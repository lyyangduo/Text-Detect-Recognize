import os
import os.path as osp 
import torch
from torch.autograd import Variable
import torch.utils.data as data
from utils.utils import setup_logger, print_args, strLabelConverter, lex_free_acc, lex_acc
from model import AN
from dataloader.SynthLoader import text_collate
from dataloader.SceneLoader import SceneLoader
from tensorboardX import SummaryWriter



import argparse
import numpy as np
import time
import logging



def main():
	parser = argparse.ArgumentParser(description='AN')
	parser.add_argument('--name', default='second_training_bn', type=str)
	
	## data setting 
	parser.add_argument('--root', default='/users/czhang/data/',type=str)
	parser.add_argument('--load_folder', default='/users/czhang/data/FAN/', type=str)
	parser.add_argument('--test_dataset', default='ic13', type=str)
	parser.add_argument('--load_width', default=256, type=int)
	parser.add_argument('--load_height', default=32, type=int)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--num_workers', default=32, type=int)
	parser.add_argument("--gpus", dest="gpu", default="1", type=str)
	parser.add_argument('--min_gt_len', default =3, type = int)
	parser.add_argument('--max_len', default=65, type=int)
	parser.add_argument("--cv", dest="context_vector", action = 'store_true')
	parser.add_argument('--lexicon',default = None, type = str)
	parser.add_argument('--max_ed',default = 3, type = int)
	parser.add_argument('--tbx_folder', default='/users/czhang/data/FAN/tbx', type=str)




	## model setting
	parser.add_argument('--load_epoch', default=0, type=int)
	parser.add_argument('--load_iter', default=0, type=int)

	parser.add_argument('--alphabet', default=' 0123456789abcdefghijklmnopqrstuvwxyz', type=str)
	## output setting
	parser.add_argument('--out_dir', default='/users/czhang/data/FAN/', type=str)

	args = parser.parse_args()

	args.nClasses = len(args.alphabet)
	args.load_folder = osp.join(args.load_folder ,args.name)
	args.out_dir = osp.join(args.out_dir ,args.name,'tests')
	if not osp.exists(args.out_dir):
		os.mkdir(args.out_dir)

	tbx_dir =osp.join(args.tbx_folder,args.name,'tests')
	if osp.exists(args.tbx_folder) == False:
		os.mkdir(args.tbx_folder)

	if osp.exists(tbx_dir) == False:
		os.mkdir(tbx_dir)

	writer = SummaryWriter(tbx_dir)

	log_path = os.path.join(args.out_dir, args.test_dataset + '.txt')

	setup_logger(log_path)

	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	device = torch.device("cuda:0")

	logging.info('model will be evaluated on %s'%(args.test_dataset))
	testset =  SceneLoader(args, args.test_dataset, False)
	logging.info('%d test samples'%(testset.__len__()))
	test_loader = data.DataLoader(testset, args.batch_size, num_workers=args.num_workers,
								  shuffle=False, pin_memory=True)

	## model
	net = AN(args)
	net = torch.nn.DataParallel(net).to(device)
	checkpoint = str(args.load_epoch)+'_'+str(args.load_iter)+'.pth'

	load_file= torch.load(osp.join(args.load_folder,checkpoint))
	net.load_state_dict(load_file['model_state_dict'])

	#net.load_state_dict(torch.load(load_file))
	net.eval()
	n_correct = 0
	skip_counter = 0
	converter = strLabelConverter(args.alphabet)

	for index, sample in enumerate(test_loader):
		imgs, gt_strs, lexicon50, lexicon1k, lexiconfull,img_paths = sample

		gt_str = gt_strs[0]
		if args.test_dataset != 'iiit5k':
			if len(gt_str) < args.min_gt_len or not gt_str.isalnum():
				print('skipping: %s'%gt_str)
				skip_counter +=1
				continue 
		else: 
			if not gt_str.isalnum():
				print('skipping: %s'%gt_str)
				skip_counter +=1
				continue 
		imgs = Variable(imgs).cuda()
		gt_ind,_ = converter.encode(gt_str)
		gt_ind = torch.IntTensor((gt_ind + [0] * args.max_len)[:args.max_len])
		preds = net(imgs,gt_ind)

		if args.lexicon is None:
			correct,pred_str,_= lex_free_acc(preds,gt_ind,converter)
			pred_lex=[]

		# lexicon decoding 
		if args.lexicon is not None:
			if args.lexicon == '50':lexicon = lexicon50
			if args.lexicon == '1k':lexicon = lexicon1k
			if args.lexicon == 'full':lexicon = full_lexicon
			correct,pred_str = lex_acc(args,lexicon,preds,gt_str,converter)
		## decode
		if correct ==0:
			writer.add_image('test_im',imgs[0,:,:,:].unsqueeze(0),index)
			writer.add_text('pred',pred_str,index)
			writer.add_text('gt',gt_str,index)

			logging.info('pred: %s gt:%s '%(pred_str, gt_str))
		n_correct += correct

	acc = n_correct*1.0/(testset.__len__()-skip_counter)
	print(testset.__len__()-skip_counter)
	logging.info('accuracy=%f'%(acc))




if __name__ == '__main__':
	main()