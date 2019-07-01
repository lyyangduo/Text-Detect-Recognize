import os
import os.path as op
import sys
import cv2
import random
from PIL import Image, ImageFilter
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
import argparse
import torchvision.transforms as transforms
from imgaug import augmenters as iaa



class SynthLoader(data.Dataset):
	def __init__(self, args, dataset, converter, is_training=True, aug=False):
		self.root = args.root
		self.dataset = dataset
		self.args = args
		self.converter = converter
		self.is_training = is_training
		self.aug = aug
		self.broken_list = ["./1817/2/363_actuating_904.jpg","./2025/2/364_SNORTERS_72304.jpg","./173/2/358_BURROWING_10395.jpg",
		"./2852/6/60_TOILSOME_79481.jpg","./2069/4/192_whittier_86389.jpg","./913/4/231_randoms_62372.jpg","./368/4/232_friar_30876.jpg","./2911/6/77_heretical_35885.jpg",
		"./495/6/81_MIDYEAR_48332.jpg","./1863/4/223_Diligently_21672.jpg"]
		self.trans1 = transforms.Compose([
					transforms.ColorJitter(brightness=0.25 ,contrast =0.5 ,saturation=0.5 ,hue =0.2),
					])

		self.trans2 = iaa.Sequential([
				iaa.Sometimes(0.5,iaa.CropAndPad(px=((-3, 3), (-10, 10), (-3, 3), (-10, 10)),
								pad_mode='edge')), # top right bottom left
				iaa.Sometimes(0.03,iaa.GaussianBlur(sigma=(0, 1))), # blur images with a sigma of 0 to 2.0
				iaa.Sometimes(0.6,iaa.AdditiveGaussianNoise(scale=(0,0.02*255))),
				iaa.Sometimes(0.05,iaa.AverageBlur(k=((4, 8), (2, 4)))),
				#iaa.Sometimes(0.4,iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.9, 1.2))),
				iaa.Sometimes(0.5,iaa.Add((-20,-20),per_channel=0.5))
				])

		self.multipler_dict = {1:382, 2:34, 3: 9,
		 4:3,  5:2,  6:1,  7:1,
		 8:1,  9:1,  10: 1, 11: 2,
		 12:3, 13: 6, 14: 8}
		self.RA = args.repeated_aug
		self.get_all_samples()


	def parse_samples(self, img_list, lexicon):
		res_imgs = []
		res_labels = []
		for line in img_list:
			parts = line.strip().split()
			if parts[0] in self.broken_list: continue
			label = lexicon[int(parts[-1])]
			#if len(label)<3: continue

			if self.aug and len(label)<15:
				multiplier = self.multipler_dict[len(label)]
				for _ in range(multiplier):
					res_imgs.append(parts[0])
					res_labels.append(lexicon[int(parts[-1])])
			else:
				res_imgs.append(parts[0])
				res_labels.append(lexicon[int(parts[-1])])

		return res_imgs, res_labels



	def get_all_samples(self):
		## check datasets
		assert (self.dataset == 'synthtext')
		self.lexicon = [x.strip() for x in open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'lexicon.txt')).readlines()]
		if self.is_training:
			self.train_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_train.txt')).readlines()
			self.train_imgs, self.train_labels = self.parse_samples(self.train_list, self.lexicon)
			self.val_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_val.txt')).readlines()
			self.val_imgs, self.val_labels = self.parse_samples(self.val_list, self.lexicon)

			self.test_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_test.txt')).readlines()
			self.test_imgs, self.test_labels = self.parse_samples(self.test_list, self.lexicon)

			self.image_paths = self.train_imgs + self.val_imgs +self.test_imgs
			self.image_labels = self.train_labels + self.val_labels + self.test_labels
	
	def vis_gt(self, image, label, save_path):
		if os.path.exists(save_path) == False:
			os.mkdir(save_path)
		im = image.numpy().astype(np.uint8).transpose((1, 2, 0))
		save_name = os.path.join(save_path, label + '.jpg')
		Image.fromarray(im).save(save_name)


			

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image_path = op.join(self.root,  'mnt/ramdisk/max/90kDICT32px', self.image_paths[index])
		image_label = self.image_labels[index]
		# img = imread(image_path, mode='RGB')
		try:
			img = Image.open(image_path)
		except Exception as e:
			logging.info(e.filename)
			img = Image.open(op.join(self.root,  'mnt/ramdisk/max/90kDICT32px', self.image_paths[index-1]))
			
		width, height = self.args.load_width, self.args.load_height
		img_resized0 = img.resize((width, height))
		if not self.aug:

			img_resized = np.array(img_resized0).astype(np.float32)
			img_resized = torch.from_numpy(img_resized.transpose((2, 0, 1)))
			imgs = [img_resized]

		if self.aug:
			imgs = []
			radius = np.int(np.abs(np.random.randn()))*3
			for _ in range(self.RA):
				img_resized = img_resized0.filter(ImageFilter.GaussianBlur(radius))
				img_resized = self.trans1(img_resized)
				img_resized = [np.array(img_resized).astype(np.uint8)]
				img_resized = self.trans2.augment_images(img_resized)[0].astype(np.float32)
				img_resized = torch.from_numpy(img_resized.transpose((2, 0, 1)))
				imgs.append(img_resized)

		text, length = self.converter.encode(image_label)

		text = torch.IntTensor((text + [0] * self.args.max_len)[:self.args.max_len])

		return imgs, text, image_path


def text_collate(batch):
	imgs = []
	labels = []
	paths = []
	for sample in batch:
		for ind in range(len(sample[0])):
			imgs.append(sample[0][ind])
			labels.append(sample[1])
			paths.append(sample[2])
	imgs = torch.stack(imgs, 0)
	labels = torch.cat(labels, 0)
	return imgs, labels, paths
