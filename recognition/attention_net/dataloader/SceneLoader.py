import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import math
import _pickle as cp

from utils.utils import strLabelConverter



class SceneLoader(data.Dataset):
	def __init__(self, args, dataset, converter):
		self.args = args
		self.root = '/users/czhang/data/'
		self.dataset = dataset
		self.img_lists = []
		self.gts = []
		self.lexicon50 = []
		self.lexicon1k = []
		self.full_lexicon = []
		self.max_len = args.max_len
		if self.dataset == 'ic03':
			self.root = self.root + '/proc/IC03/' 
			self.gt_file = self.root + '/gt.cp'
			gt = cp.load(open(self.gt_file,'rb'))
			self.img_lists = gt['imname']
			self.gts = gt ['gt']
			self.lexicon50 = gt['sample_lex']['50']
			self.lexicon1k = None
			self.full_lexicon = gt['dset_lex']['full']

		elif self.dataset == 'ic13':
			self.root = self.root + '/proc/IC13/' 
			self.gt_file = self.root + '/gt.cp'
			gt = cp.load(open(self.gt_file,'rb'))
			self.img_lists = gt['imname']
			self.gts = gt ['gt']
			self.lexicon50 = None
			self.lexicon1k = None
			self.full_lexicon = None

		elif self.dataset == 'iiit5k':
			self.root = self.root + '/proc/IIIT5K/' 
			self.gt_file = self.root + '/gt.cp'
			gt = cp.load(open(self.gt_file,'rb'))
			self.img_lists = gt['imname']
			self.gts = gt ['gt']
			self.lexicon50 = gt['sample_lex']['50']
			self.lexicon1k = gt['sample_lex']['1k']
			self.full_lexicon = None

		elif self.dataset == 'svt':
			self.root = self.root + '/proc/SVT/' 
			self.gt_file = self.root + '/gt.cp'
			gt = cp.load(open(self.gt_file,'rb'))
			self.img_lists = gt['imname']
			self.gts = gt ['gt']
			self.lexicon50 = gt['sample_lex']['50']
			self.lexicon1k = None
			self.full_lexicon = None       
		else:
			print('unknown dataset!!!')
			exit()
	
	def __getitem__(self, index):
		return  self.pull_item(index)

	def __len__(self):
		return len(self.img_lists)

   

	def pull_item(self, index):
		img_path = osp.join(self.root,'img',self.img_lists[index])
		img = cv2.imread(img_path)
		width, height = self.args.load_width, self.args.load_height
		img = cv2.resize(img.copy(), (width, height))
		img = img[:, :, (2, 1, 0)] ## rgb
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

		gt = self.gts[index].lower()
		# gt = gt.replace('\'','').replace('!','').replace('&','')
		# gt,_ = self.converter.encode(gt)
		# gt = torch.IntTensor((gt + [0] * self.max_len)[:self.max_len])
		if self.lexicon50 is not None:
			lexicon50 = self.lexicon50[index]
		else: lexicon50 = []

		if self.lexicon1k is not None:
			lexicon1k = self.lexicon1k[index]
		else: lexicon1k = []

		if self.full_lexicon is not None:
			lexiconfull = self.full_lexicon
		else: lexiconfull = []
		return torch.from_numpy(img).permute(2, 0, 1).float(), gt, lexicon50, lexicon1k, lexiconfull, img_path


# def samples_collate(batch):
#     imgs = []
#     segs = []
#     for sample in batch:
#         imgs.append(sample[0])
#         segs.append(sample[1])

#     return torch.stack(imgs, 0), torch.stack(segs, 0)