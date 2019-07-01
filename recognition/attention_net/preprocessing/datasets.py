"""
Author: Ankush Gupta
Date: 27 September, 2017.

Interface to various text-recognition datasets.
NOTE: Only test-sets supported ATM.
"""

import _pickle as cp
import numpy as np
import os
import os.path as osp
from PIL import Image
import shutil
import re
import glob
import xml.etree.ElementTree as ET
import scipy.io as sio



from utils import bb_utils as bbu
from utils import im_utils as imu
from utils.etree_to_dict import etree_to_dict

class IC03(object):

  def __init__(self, dir_prefix='/users/czhang/', name='IC03', subset='test', pad=0.05):
    self.name = name
    self.subset = subset
    self.pad = pad
    # directory where the raw dataset information is stored:
    self.data_dir = osp.join(dir_prefix, 'data/raw/IC03/')
    # directory where the processed data is stored:
    self.proc_dir = osp.join(dir_prefix, 'data/proc/IC03')

    if osp.exists(self.proc_dir) == False:
      os.mkdir(self.proc_dir)
    # load-in the dataset:
    self.__read_data()

  def __read_data(self):
    """
    Loads into memory the ground-truth information.
    """
    f = osp.join(self.data_dir, 'icdarWholeTestStruct.pickle')
    gt_dat = cp.load(open(f,'rb'),encoding='latin1')

    # get the scene-level image names:
    self.imnames = [imname.decode() for imname in gt_dat['testStruct']['imgname']]
    n_ims = len(self.imnames)
    # get the word info in each scene-image:
    word_info = gt_dat['testStruct']['bbox']
    assert len(word_info)==n_ims, 'number of images != word_info'
    self.word_lex = {'50': []}
    self.scene_im_id = []
    self.word_gt = []
    self.word_xywh = []
    for i in range(n_ims):
      i_imname = self.imnames[i]
      i_words = word_info[i]
      for k in i_words.keys(): i_words[k] = np.atleast_1d(i_words[k])
      n_words = i_words['x'].shape[0]
      # read the lexicon for this scene-image:
      lex_fname = osp.join(self.data_dir,'lex50','I%05d.jpg.txt'%i)
      with open(lex_fname, 'r') as f:
        i_lex = f.readlines()
      i_lex = [w.strip() for w in i_lex]
      for j in range(n_words):
        self.scene_im_id.append(i)
        self.word_lex['50'].append(i_lex)
        if isinstance(i_words['trueTag'][j],(bytes, bytearray)):
          if 'xa3' in str(i_words['trueTag'][j]): 
            aaa =  str(i_words['trueTag'][j])
            aaa = aaa[4:-1]
            self.word_gt.append(aaa)
          else:
            self.word_gt.append(i_words['trueTag'][j].decode())
        elif isinstance(i_words['trueTag'][j],str):
          self.word_gt.append(i_words['trueTag'][j])
        self.word_xywh.append([ int(i_words['x'][j]), int(i_words['y'][j]),
                                int(i_words['w'][j]), int(i_words['h'][j]) ])
    # read in the global / dataset-level lexicons:
    dset_lex_names = {'full':'lexFull.txt', '50k':'lex50k.txt'}
    self.dset_lex = {}
    for lex in dset_lex_names.keys():
      fname =  osp.join(self.data_dir, dset_lex_names[lex])
      with open(fname, 'r') as f:
        lex_words = f.readlines()
      lex_words = [w.strip() for w in lex_words]
      self.dset_lex[lex] = lex_words

  def save(self):
    """
    Saves the dataset in the "standard" format.
    """
    if not osp.exists(self.proc_dir):
      os.makedirs(self.proc_dir)
    img_dir = osp.join(self.proc_dir, 'img')
    if not osp.exists(img_dir):
      os.makedirs(img_dir)
    # extract image-patches, and save:
    n_words = len(self.word_gt)
    print('Saving %d word-crops...'%n_words)
    self.word_imnames = []
    for i in range(n_words):
      scene_im = osp.join(self.data_dir, self.imnames[self.scene_im_id[i]])
      im = Image.open(scene_im)
      word_coord = bbu.xywh2coords(np.array(self.word_xywh[i]).reshape([1,4]))
      word_crop = imu.extract_patch(im, word_coord, pad=self.pad)
      i_name = '%05d.jpg'%i
      self.word_imnames.append(i_name)
      crop_imname = osp.join(img_dir, i_name)
      word_crop.save(crop_imname)

    # build the word-crop gt structure:
    gt = {'dset': self.name,
          'imname': self.word_imnames,
          'gt': self.word_gt,
          'sample_lex': self.word_lex,
          'dset_lex': self.dset_lex}
    with open(osp.join(self.proc_dir, 'gt.cp'), 'wb') as f:
      cp.dump(gt, f)
    print('Wrote ground-truth.')


class IC13(object):
  """ICDAR 2013 word-recognition dataset."""
  def __init__(self, dir_prefix='/users/czhang/', name='IC13', subset='test'):
    self.name = name
    self.subset = subset
    # directory where the raw dataset information is stored:
    self.data_dir = osp.join(dir_prefix, 'data/raw/IC13')
    # directory where the processed data is stored:
    self.proc_dir = osp.join(dir_prefix, 'data/proc/IC13')
    # image-dir:
    if subset == 'test':
      self.im_dir = osp.join(self.data_dir, 'Challenge2_Test_Task3_Images')
    elif subset == 'train':
      self.im_dir = osp.join(self.data_dir, 'Challenge2_Training_Task3_Images_GT')
    else:
      raise ValueError('Subset %s not understood.'%subset)
    # load-in the dataset:
    self.__read_data()

  def __read_data(self, ):
    """Reads the ground-truth data."""
    # get in the list of all images:
    self.imnames = [osp.basename(f) for f in glob.glob(osp.join(self.im_dir, '*.png'))]
    # read the gt annotation:
    self.ann = {}
    with open(osp.join(self.data_dir, 'gt-%s.txt'%self.subset), 'r') as f:
      ann = f.readlines()
    for i in range(len(ann)):
      imname, word = re.match('(.*), (.*)', ann[i]).groups()
      imname = osp.basename(imname)
      word = word.strip().strip('"')
      self.ann[imname] = word
    # make sure that there is annotation all for all word-images:
    for f in self.imnames:
      if f not in self.ann.keys():
        raise Exception('Image %s does not have a word annotation.'%f)

  def save(self):
    im_out_dir = osp.join(self.proc_dir, 'img')
    if not osp.exists(im_out_dir):
      os.makedirs(im_out_dir)
    # copy over the word patches:
    self.word_imnames, self.word_gt = [], []
    for f in self.imnames:
      shutil.copy2(osp.join(self.im_dir, f), im_out_dir)
      self.word_imnames.append(osp.join(im_out_dir, f))
      self.word_gt.append(self.ann[f])
    # build the word-crop gt structure:
    gt = {'dset': self.name,
          'imname': self.word_imnames,
          'gt': self.word_gt,
          'sample_lex': [],
          'dset_lex': []}
    with open(osp.join(self.proc_dir, 'gt.cp'), 'wb') as f:
      cp.dump(gt, f)
    print('Wrote ground-truth.')


class SVT(object):
  def __init__(self, dir_prefix='/users/czhang/', name='SVT', subset='test', pad=0.05):
    self.name = name
    self.subset = subset
    self.pad = pad
    # directory where the raw dataset information is stored:
    self.data_dir = osp.join(dir_prefix, 'data/raw/SVT')
    # directory where the processed data is stored:
    self.proc_dir = osp.join(dir_prefix, 'data/proc/SVT')
    # image-dir:
    self.im_dir = osp.join(self.data_dir, 'img')
    self.gt_fname = osp.join(self.data_dir, '%s.xml'%subset)
    if subset not in ['test', 'trian']:
      raise ValueError('Subset %s not understood.'%subset)
    # load-in the dataset:
    self.__read_data()

  def __read_data(self):
    """
    Loads into memory the ground-truth information.
    """
    # convert xml to dict:
    gt_xml = ET.parse(self.gt_fname).getroot()
    gts = etree_to_dict(gt_xml)['tagset']['image']

    gt = {} # place-holder for the output
    n_ims = len(gts)

    xywh_ks = ['@x','@y','@width','@height']
    self.dset_lex = {}
    self.word_lex = {'50': []}
    self.imnames = []
    self.scene_im_id = []
    self.word_gt = []
    self.word_xywh = []
    for i in range(n_ims):
      i_gt = gts[i]
      # get the scene-level info:
      scene_imname = osp.basename(i_gt['imageName'])
      self.imnames.append(scene_imname)
      i_lex = [w for w in i_gt['lex'].split(',')]
      # parse the word-level info:
      gt_bbs = i_gt['taggedRectangles']['taggedRectangle']
      if not isinstance(gt_bbs, list): gt_bbs = [gt_bbs]
      bbs, txt = [], []
      for j, gt_bb in enumerate(gt_bbs):
        self.scene_im_id.append(i)
        self.word_xywh.append([int(gt_bb[k]) for k in xywh_ks])
        self.word_gt.append(gt_bb['tag'])
        self.word_lex['50'].append(i_lex)

  def save(self):
    """
    Saves the dataset in the "standard" format.
    """
    out_im_dir = osp.join(self.proc_dir, 'img')
    if not osp.exists(out_im_dir): os.makedirs(out_im_dir)
    # extract image-patches, and save:
    n_words = len(self.word_gt)
    print('Saving %d word-crops...'%n_words)
    self.word_imnames = []
    for i in range(n_words):
      scene_im = osp.join(self.im_dir, self.imnames[self.scene_im_id[i]])
      im = Image.open(scene_im)
      word_coord = bbu.xywh2coords(np.array(self.word_xywh[i]).reshape([1,4]))
      word_crop = imu.extract_patch(im, word_coord, pad=self.pad)
      i_name = '%05d.jpg'%i
      self.word_imnames.append(i_name)
      crop_imname = osp.join(out_im_dir, i_name)
      word_crop.save(crop_imname)
    # build the word-crop gt structure:
    gt = {'dset': self.name,
          'imname': self.word_imnames,
          'gt': self.word_gt,
          'sample_lex': self.word_lex,
          'dset_lex': self.dset_lex}
    with open(osp.join(self.proc_dir, 'gt.cp'), 'wb') as f:
      cp.dump(gt, f)
    print('Wrote ground-truth.')


class IIIT5K(object):
  """IIIT5K word-recognition dataset."""
  def __init__(self, dir_prefix='/users/czhang/', name='IIIT5K', subset='test'):
    self.name = name
    self.subset = subset
    # directory where the raw dataset information is stored:
    self.data_dir = osp.join(dir_prefix, 'data/raw/IIIT5K')
    # directory where the processed data is stored:
    self.proc_dir = osp.join(dir_prefix, 'data/proc/IIIT5K')
    if subset not in ['train', 'test']:
      raise ValueError('Subset %s not understood.'%subset)
    self.im_dir = osp.join(self.data_dir, subset)
    self.gt_fname = osp.join(self.data_dir, '%sdata.mat'%subset)
    self.__read_data()

  def __read_data(self, ):
    """Reads the ground-truth data."""
    gtdat = sio.loadmat(self.gt_fname)
    gtdat = np.squeeze(gtdat['%sdata'%self.subset])
    n_ims = len(gtdat)
    self.imnames = []
    self.word_gt = []
    self.word_lex = {'50': [], '1k':[]}
    self.dset_lex = {}
    for i in range(n_ims):
      i_name, i_ann, i_lex50, i_lex1k = gtdat[i]
      i_name = osp.basename(str(i_name[0]))
      i_ann = str(i_ann[0])
      i_lex50, i_lex1k = np.squeeze(i_lex50), np.squeeze(i_lex1k)
      i_lex50 = [str(w[0]) for w in i_lex50]
      i_lex1k = [str(w[0]) for w in i_lex1k]
      self.imnames.append(i_name)
      self.word_gt.append(i_ann)
      self.word_lex['50'].append(i_lex50)
      self.word_lex['1k'].append(i_lex1k)

  def save(self):
    im_out_dir = osp.join(self.proc_dir, 'img')
    if not osp.exists(im_out_dir): os.makedirs(im_out_dir)
    # copy over the word patches:
    self.word_imnames = []
    for f in self.imnames:
      shutil.copy2(osp.join(self.im_dir, f), im_out_dir)
      self.word_imnames.append(f)
    # build the word-crop gt structure:
    gt = {'dset': self.name,
          'imname': self.word_imnames,
          'gt': self.word_gt,
          'sample_lex': self.word_lex,
          'dset_lex': self.dset_lex}
    with open(osp.join(self.proc_dir, 'gt.cp'), 'wb') as f:
      cp.dump(gt, f)
    print('Wrote ground-truth.')

if __name__ == '__main__':
  # dsets = [ IIIT5K, IC03, IC13, SVT]
  dsets = [IC03]
  for dset in dsets:
    dset = dset()
    dset.save()
    print('dataset: %s :: done'%dset.name)
    print('='*40)

