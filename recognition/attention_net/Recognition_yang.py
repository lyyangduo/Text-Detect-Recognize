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
import tarfile
from pathlib import Path
import tqdm
import shutil
import glob
import time

parser = argparse.ArgumentParser(description='AN')
parser.add_argument('--load_width', default=256, type=int)
parser.add_argument('--load_height', default=32, type=int)
parser.add_argument("--gpus", dest="gpu", default="1", type=str)
parser.add_argument('--max_len', default=65, type=int)
parser.add_argument("--cv", dest="context_vector", action = 'store_true')
parser.add_argument('--alphabet', default=' 0123456789abcdefghijklmnopqrstuvwxyz', type=str)
parser.add_argument('--detection_path', default='', type=str)


args, unknown = parser.parse_known_args()




args.nClasses = len(args.alphabet)
width, height = args.load_width, args.load_height
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device("cuda:0")
net = AN(args)
net = torch.nn.DataParallel(net).to(device)
checkpoint = '../attention_net/0_480000.pth'
load_file= torch.load(checkpoint)
net.load_state_dict(load_file['model_state_dict'])
net.eval()
converter = strLabelConverter(args.alphabet)

pickle_path = args.detection_path
with open (pickle_path,'rb') as p:
    data=pickle.load(p)
    
text_result={}
frame_paths = list(data.keys())




for image_name in frame_paths:
    img_path = image_name
    all_boxes = data[img_path]
    img_raw = cv2.imread(img_path)
    temp_key =img_path
    
    text_result[temp_key]=[]


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
        if conf_score>0.8:
            text_result[temp_key].append(pred_str)
            #print (pred_str)   # uncomment if you want to see predict strings!!
            #print (pred_str)

 
        
save_pickle_path ='Recognition.pkl'
with open (save_pickle_path,'wb') as f:
    pickle.dump(text_result, f)