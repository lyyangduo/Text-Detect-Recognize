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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pickle
import cv2

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
parser.add_argument("--gpus", dest="gpu", default="2", type=str)
parser.add_argument('--min_gt_len', default =3, type = int)
parser.add_argument('--max_len', default=65, type=int)
parser.add_argument("--cv", dest="context_vector", action = 'store_true')
parser.add_argument('--lexicon',default = None, type = str)
parser.add_argument('--max_ed',default = 3, type = int)
parser.add_argument('--tbx_folder', default='/scratch/shared/slow/yangl/code/attention_net/tbx/', type=str)




## model setting
parser.add_argument('--load_epoch', default=0, type=int)
parser.add_argument('--load_iter', default=0, type=int)

parser.add_argument('--alphabet', default=' 0123456789abcdefghijklmnopqrstuvwxyz', type=str)
## output setting
parser.add_argument('--out_dir', default='/scratch/shared/slow/yangl/code/attention_net/output/', type=str)
args, unknown = parser.parse_known_args()
print (args)



args.nClasses = len(args.alphabet)
args.load_folder = osp.join(args.load_folder ,args.name)
#args.out_dir = osp.join(args.out_dir ,args.name,'tests')
#args.out_dir = osp.join(args.out_dir ,args.name,'tests')
if not osp.exists(args.out_dir):
    os.mkdir(args.out_dir)

#tbx_dir =osp.join(args.tbx_folder,args.name,'tests')
tbx_dir =args.tbx_folder
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


net = AN(args)
net = torch.nn.DataParallel(net).to(device)
checkpoint = '../attention_net/0_480000.pth'

load_file= torch.load(checkpoint)
net.load_state_dict(load_file['model_state_dict'])

#net.load_state_dict(torch.load(load_file))
net.eval()
n_correct = 0
skip_counter = 0
converter = strLabelConverter(args.alphabet)



pickle_path ='/scratch/shared/slow/yangl/code/textinvideo/pixel_link/file_test_box.pkl'
with open (pickle_path,'rb') as p:
    data=pickle.load(p)
    
text_result={}
all_keys= list(data.keys())
for kk in range (len(all_keys)):
    
    temp_key =all_keys[kk].split('/')[-2] 
    text_result[temp_key]=[]

#all_keys= list(data.keys())[261319:261379]  #1002
#all_keys= list(data.keys())[1627:1686]   #5499
#all_keys= list(data.keys())[13429:13514]   #2517
width, height = args.load_width, args.load_height
progress = 0
for img_id in range (len(all_keys))    :
    img_path = all_keys[img_id]
    all_boxes = data[img_path]
    #temp_box = all_boxes[9]

    img_raw = cv2.imread(img_path)
    temp_key =img_path.split('/')[-2] 
    #print (temp_key)
    for kk in range (len(all_boxes)):

        temp_box = all_boxes[kk]
        odd= np.asarray([temp_box[0], temp_box[2],temp_box[4], temp_box[6]])
        even= np.asarray([temp_box[1], temp_box[3],temp_box[5], temp_box[7]])

        y_min = min(odd)
        y_max = max(odd)

        x_min = min(even)
        x_max = max(even)


        new_img = img_raw.copy()
        new_img=new_img[x_min:x_max,y_min:y_max, :]


        img = cv2.resize(new_img.copy(), (width, height))
        img = img[:, :, (2, 1, 0)] ## rgb
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        imgs = torch.from_numpy(img).permute(2, 0, 1).float()
        imgs= imgs.view(1, 3,32,256)
        imgs = Variable(imgs).cuda()
        gt_ind,_ = converter.encode('abc')
        gt_ind = torch.IntTensor((gt_ind + [0] * args.max_len)[:args.max_len])
        preds = net(imgs,gt_ind)

        preds_label = torch.argmax(preds,dim =2)
        preds_conf = torch.max(preds,dim =2)

        word_len = torch.sum(preds_label!=0)
        conf_score = torch.mean(preds_conf[0][0][0:word_len])

        _,pred_str,_= lex_free_acc(preds,gt_ind,converter)
        #print (pred_str)
        #print (conf_score.cpu().detach().numpy())
        if conf_score>0.95:
            text_result[temp_key].append(pred_str)
            #print (pred_str)
            
        if progress%1000 ==0 :
            print (progress)
        progress=progress+1
        
        
        
final_keys = list(text_result.keys())
for temp_key in final_keys:
    text_result[temp_key]= list(set(text_result[temp_key]))
#     if len(list(set(text_result[temp_key])))>0:
#         print (temp_key)

        
save_pickle_path ='/scratch/shared/slow/yangl/code/textinvideo/MSR_VTT_test_text.pkl'
with open (save_pickle_path,'wb') as f:
    pickle.dump(text_result, f)