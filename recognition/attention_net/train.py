import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from dataloader.SynthLoader import SynthLoader,text_collate
from dataloader.SceneLoader import SceneLoader
from utils.utils import setup_logger, print_args, strLabelConverter,lex_free_acc,mask
from tensorboardX import SummaryWriter
from utils.loss import CenterLoss

from model import AN
import numpy as np
import time
import logging


def main():
    parser = argparse.ArgumentParser(description='AN')
    parser.add_argument('--name', default='bn_smaller_batch', type=str)
    
    ## data setting 
    parser.add_argument('--root', default='/scratch/local/ssd/datasets',type=str)
    parser.add_argument('--train_dataset', default='synthtext', type=str)
    parser.add_argument('--test_dataset', default='ic03', type=str)
    parser.add_argument('--vis_gt', default=False, type=bool)
    parser.add_argument('--vis_gt_path', default='/users/czhang/data/vis', type=str)
    parser.add_argument('--load_width', default=256, type=int)
    parser.add_argument('--load_height', default=32, type=int)
    parser.add_argument("--gpus", dest="gpu", default="0", type=str)
    parser.add_argument('--min_gt_len', default =3, type = int)
    parser.add_argument("--aug", dest="aug", action = 'store_true')
    parser.add_argument("--RA", dest="repeated_aug",default ='1',type=int)


    ## model setting
    parser.add_argument('--alphabet', default=' 0123456789abcdefghijklmnopqrstuvwxyz', type=str)
    #parser.add_argument('--ignore_case', default=True, type=bool)
    parser.add_argument('--max_len', default=65, type=int)
    parser.add_argument("--cv", dest="context_vector", action = 'store_true')

    ## optim setting
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--resume_i', default=0, type=int)
    parser.add_argument('--resume_j', default=0, type=int)

    parser.add_argument('--cl_weight', default=1, type=int, help='center loss weight')
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--optim', default='adadelta', type=str, help='sgd, adam, adadelta')
    # parser.add_argument('--clip_grad', default=False, type=bool)
    parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--max_epoches', default=1000, type=int)
    # parser.add_argument('--adjust_lr', default='800, 1600', type=str)

    ## output setting
    parser.add_argument('--log_iter', default=10, type=int)
    parser.add_argument('--eval_iter', default=2500,type=int )
    parser.add_argument('--save_iter', default=2500, type=int)
    parser.add_argument('--save_folder', default='/users/czhang/data/FAN/', type=str)
    parser.add_argument('--tbx_folder', default='/users/czhang/data/FAN/tbx', type=str)

    parser.add_argument('--eval_vis_num', default=15, type=int)
    parser.add_argument('--max_iter', default=2000000, type=int)
    
    
    args = parser.parse_args()
    args.save_folder = osp.join(args.save_folder ,args.name)
    if osp.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)

    tbx_dir =osp.join(args.tbx_folder,args.name)
    if osp.exists(args.tbx_folder) == False:
        os.mkdir(args.tbx_folder)

    if osp.exists(tbx_dir) == False:
        os.mkdir(tbx_dir)
    writer = SummaryWriter(tbx_dir)
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    ##
    args.nClasses = len(args.alphabet)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda:0")


    setup_logger(log_file_path)
    print_args(args)
    torch.set_default_tensor_type('torch.FloatTensor')

    ## setup converter
    converter = strLabelConverter(args.alphabet)

    ## setup dataset
    logging.info('model will be trained on %s'%(args.train_dataset))
    trainset =  SynthLoader(args, args.train_dataset, converter, aug = args.aug)
    logging.info('%d training samples'%(trainset.__len__()))
    train_loader = data.DataLoader(trainset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True,  collate_fn=text_collate, pin_memory=True)

    logging.info('model will be evaluated on %s'%(args.test_dataset))
    testset =  SceneLoader(args, args.test_dataset, False)
    logging.info('%d test samples'%(testset.__len__()))
    test_loader = data.DataLoader(testset, 1, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)

    ## setup model
    net = AN(args)
    net = torch.nn.DataParallel(net).to(device)
    centers = None


    if args.resume_i !=0 or args.resume_j!=0:
        resume_file = osp.join(args.save_folder,str(args.resume_i)+'_'+str(args.resume_j)+'.pth')
        logging.info('Resuming training, loading {}...'.format(resume_file))
        checkpoint = torch.load(resume_file)
        #net.load_state_dict(checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        centers = checkpoint['class_centers']

    ## setup criterion
    criterion = nn.CrossEntropyLoss()
    criterion2 = CenterLoss(device,centers)

    ## setup optimizer
    if args.cl_weight != 0:
        parameters =  list(net.parameters()) + list(criterion2.parameters())
    else:
        parameters = net.parameters()

    if args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        logging.info('model will be optimed by sgd')
    elif args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        logging.info('model will be optimed by adam')
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(parameters,lr=args.lr, weight_decay=args.weight_decay)
        logging.info('model will be optimed by adadelta')
    else:
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        logging.info('model will be optimed by adam')


    ## train model
    cudnn.benchmark = True
    net.train()
    iter_counter = args.resume_j+1
    acc_max = 0
    running_loss,running_cenloss,running_croloss = 0.,0.,0.

    for i in range(args.max_epoches):
        i = args.resume_i +i 
        t0 = time.time()
        for j, batch_samples in enumerate(train_loader):
            j = args.resume_j + j+1
            imgs, labels, paths = batch_samples
            imgs = Variable(imgs.float()).to(device)
            labels = Variable(labels.long()).to(device) #[batch*len]
            if args.context_vector or args.cl_weight !=0:
                preds,gts = net(imgs,labels) #[batch,len,classes]
                masks = mask(args,labels.view(args.batch_size,args.max_len),device)
                center_loss = criterion2(gts,labels,masks)
                running_cenloss += center_loss.item()

            else:
                preds = net(imgs,labels)
                center_loss = 0

            ce_loss = criterion(preds.view(-1, args.nClasses), labels.view(-1))
            loss = ce_loss + 0.01* args.cl_weight*center_loss

            optimizer.zero_grad()
            loss.backward()
            if args.cl_weight != 0:
                for param in criterion2.parameters():
                    # update class centers
                    # remove the effect of lambda on updating centers
                    # lr of center loss set to 0.5 of the model lr 
                    param.grad.data *= (0.5 / (0.01*args.cl_weight))

            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()
            running_loss += loss.item()
            running_croloss += ce_loss.item()

            if iter_counter % args.log_iter == 0:
                t1 = time.time()
                acc,pred_samples,label_samples = lex_free_acc(preds,labels,converter)
                print('epoch:%3d  iter:%6d  loss:%4.6f  acc:%4.6f  %4.6fs/batch'%(i, j, running_loss/args.log_iter,acc, (t1-t0)/args.log_iter))
                writer.add_scalar('train/train_word_accuracy',acc,j)
                writer.add_scalar('train/train_loss',running_loss/args.log_iter,j)
                if args.cl_weight != 0:
                    writer.add_scalar('train/train_ce_loss',running_croloss/args.log_iter,j)
                    writer.add_scalar('train/train_center_loss',running_cenloss/args.log_iter,j)


                if iter_counter % (100*args.log_iter) == 0:
                    visual_img = imgs[0,:,:,:].unsqueeze(0)
                    writer.add_image('train/train_im',visual_img,j)
                    visual_txt = 'gt: '+ str(label_samples[0])+ ' ----- pred: '+str(label_samples[0])
                    writer.add_text('train/train_txt',visual_txt,j)
                t0 = time.time()
                running_loss,running_cenloss,running_croloss = 0.,0.,0.

            if iter_counter % args.save_iter == 0:
                print('Saving state, epoch: %d iter:%d'%(i, j))
                torch.save({
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'class_centers': criterion2.centers},
                        args.save_folder + '/'  + repr(i) + '_' + repr(j) + '.pth')
     
            if iter_counter % args.eval_iter == 0:
                ## eval model
                net.eval()
                n_correct = 0
                skip_counter = 0
                for index, sample in enumerate(test_loader):
                    imgs, gt_strs, lexicon50, lexicon1k, lexiconfull,img_paths = sample

                    gt_str = gt_strs[0]
                    if len(gt_str) < args.min_gt_len or not gt_str.isalnum():
                        skip_counter +=1
                        continue 
                    imgs = Variable(imgs).cuda()
                    gt_ind,_ = converter.encode(gt_str)
                    gt_ind = torch.IntTensor((gt_ind + [0] * args.max_len)[:args.max_len])
                    if args.context_vector or args.cl_weight!=0:
                        preds,_ = net(imgs,gt_ind)
                    else:
                        preds = net(imgs,gt_ind)

                    correct,pred_str,_= lex_free_acc(preds,gt_ind,converter)
                    n_correct += correct


                acc = n_correct*1.0/(testset.__len__()-skip_counter)
                if acc > acc_max:
                    acc_max = acc
                logging.info('accuracy=%f   acc_max=%f'%(acc, acc_max))
                writer.add_scalar('val/val_word_accuracy',acc,j)

                net.train()

            if iter_counter > args.max_iter:
                break
            iter_counter += 1


    torch.save(net.state_dict(), args.save_folder + '/final_0.pth')
    logging.info('The training stage on %s is over!!!' % (args.train_dataset))


if __name__ == '__main__':
    main()