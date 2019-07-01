"""
Script to export the OCR detections from GOOGLE
into word-crops for training (or fine-tuning).

Saves word-crop window images and their names
and corresponding annotations in a simple text file.
"""
import os
import random
from PIL import Image
import argparse
import numpy as np
import os.path as osp
import simplejson as json
import cPickle as cp
import matplotlib.pyplot as plt
import codecs
import read_gt

def get_mean_hw(x,y):
  """
  Get the mean width and height
  of a quadrilateral.
  """
  w1 = abs(x[1]-x[0])
  w2 = abs(x[2]-x[3])
  w = np.mean([w1, w2])

  h1 = abs(y[3]-y[0])
  h2 = abs(y[2]-y[1])
  h = np.mean([h1, h2])

  return w,h

def rotate_bb(x,y,th_deg):
  """
  Rotate the point-set by TH_DEGs
  around the center.
  """
  th = np.deg2rad(th_deg)
  c,s = np.cos(th), np.sin(th)
  R = np.array([[c, -s],[s, c]])

  xy = np.c_[np.array(x),np.array(y)]
  mu = np.mean(xy,axis=0)
  xy_c = xy-mu
  xy_r =  xy_c.dot(R.T) + mu
  return xy_r[:,0],xy_r[:,1]

def translate(x,y,d):
  """
  D : in fraction of width and height respectively.
  """
  w,h = get_mean_hw(x,y)
  d *= np.array([w,h])
  xy_t = np.c_[x,y] + d
  return xy_t[:,0],xy_t[:,1]

def scale(x,y,s):
  """
  S : in fraction of width and height respectively.
  """
  xy = np.c_[x,y]
  mu = np.mean(xy,axis=0)

  w,h = get_mean_hw(x,y)
  s *= np.array([w,h])/2.0
  xy_s = s * np.array([[-1, 1, 1, -1],[-1,-1,1,1]]).T  + mu
  return xy_s[:,0],xy_s[:,1]

def xy2aa(x,y):
  """
  Get axis-aligned x andy from
  four vertices:
  """
  x,y = np.array(x), np.array(y)
  x_aa = [np.min(x), np.max(x), np.max(x), np.min(x)]
  y_aa = [np.min(y), np.min(y), np.max(y), np.max(y)]
  return x_aa,y_aa

def jitter_window(x,y):
  """
  Jitter the window defined by the coordinates
  (x,y). Jittering is done through cropping,
  resizing, rotation, padding etc.
  """
  # sample a small rotation:
  TH_MAX = 15  # +-5 deg max rotation
  th = TH_MAX * (2*np.random.rand()-1)
  x,y = rotate_bb(x,y,th)

  # randomly jitter the location:
  D_MAX = np.array([0.05, 0.10]) # width,height max %age translation:
  d = (2*np.random.rand(1,2)-1) * D_MAX
  x,y = translate(x,y,d)

  # randomly scale the width and height:
  S_MAX = np.array([0.05,0.15])
  s = 1 + S_MAX * (2*np.random.rand()-1)
  x,y = scale(x,y,s)

  return x,y

def is_valid_window(x,y):
  """
  Check if the window defined by x and y
  coordinates is a "valid" word-window
  (primarily filter on aspect ratio).
  Assumes, that the coordinates are
  roughly in CW order.
  """
  w,h = get_mean_hw(x,y)
  asp_ratio = w / (h+1e-4)
  return asp_ratio >= 0.5

def pad_bb(xy,f_pad):
  w,h = get_mean_hw(xy[:,0],xy[:,1])
  pad = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
  pad = pad * f_pad/2.0 * np.array([w,h])[None,:]
  xy = xy+ pad
  return xy

def save_bb(bb_num,im,out_im_dir,xy,txt,f_pad,viz):
  # try:
  # extract the text window:
  w,h = get_mean_hw(xy[:,0],xy[:,1])
  imsz = tuple(np.ceil([w,h]).astype(int))
  # pad the box:
  xy = pad_bb(xy,f_pad)
  # extract the patch:
  quad = tuple(np.round(xy[[0,3,2,1],:]).flatten().astype(int))
  im_patch = im.transform(imsz,Image.QUAD,quad,Image.BILINEAR)
  w,h = im_patch.size
  min_h = 128
  if h < min_h:
    new_sz = (int(min_h * w / (h+0.0)), min_h)
    im_patch = im_patch.resize(new_sz, Image.ANTIALIAS)

  out_patch_fname = osp.join(out_im_dir,'%d.jpg'%bb_num)
  print out_patch_fname
  im_patch.save(out_patch_fname)

  if viz:
    print('annotation: '+txt)
    plt.imshow(im_patch)
    plt.show(block=True)
  return True
  # except:
  #   return False


def main(opts):
  base_dir = '/Volumes/Expanse/data_local/datasets'
  dset = {'ic13': { 'gt': osp.join(base_dir,'icdar-2013'),
                    'google': osp.join(base_dir,'icdar-2013','google_ocr'),
                    'ankush': osp.join(base_dir,'icdar-2013','ankush') },
          'ic11': { 'gt': osp.join(base_dir,'icdar-2011'),
                    'google': osp.join(base_dir,'icdar-2011','google_ocr'),
                    'ankush': osp.join(base_dir,'icdar-2011','ankush') },
          'svt': { 'gt': osp.join(base_dir,'svt'),
                    'google': osp.join(base_dir,'svt','google_ocr/test'),
                    'ankush': osp.join(base_dir,'svt','ankush') }
         }

  dset_name = opts.dset
  imname_template = 'img_%s.jpg'

  # read in the ground-truth:
  gt_dir = dset[dset_name]['gt']
  gt = read_gt.get_gt(dset_name,gt_dir,'test')
  in_im_dir = osp.join(gt_dir,'test_im')

  # create the output directory:
  out_dir = osp.join(gt_dir,'crops','pad_%.2f'%opts.f_pad)
  out_im_dir = osp.join(out_dir,'im')
  if not osp.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(out_im_dir)

  # extract padded word crops one-by-one:
  bb_num = 0 #index into image patch
  bb_gt = {} # this maps from saved bb number to gt string
  for im_id,im_gt in gt.items():
    n_words = len(im_gt[1])
    bb,txt = im_gt
    bb[2:4,:] = bb[:2,:] + bb[2:4,:]

    imname = osp.join(in_im_dir,imname_template%im_id)
    im = Image.open(imname)
    # crop each word out:
    for i in xrange(n_words):
      ibb = bb[:,i]
      x = np.array([ibb[0],ibb[2],ibb[2],ibb[0]])
      y = np.array([ibb[1],ibb[1],ibb[3],ibb[3]])

      s = save_bb(bb_num,im,out_im_dir,np.c_[x,y],txt[i],opts.f_pad,opts.viz)
      if s:
        bb_gt[str(bb_num)] = txt[i]
        bb_num += 1

  ann_fname = osp.join(out_dir,'crops_gt.cp')
  with open(ann_fname,'wb') as f:
    cp.dump(bb_gt,f)


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Extract patches from Google OCR output.')
  parser.add_argument('dset',choices=['ic13','ic11','svt'],help='name of the dataset to evaluate on')
  parser.add_argument('--fpad',type=float,default=0.00,dest='f_pad',help='fraction of padding.')
  parser.add_argument('--viz',default=False,action='store_true',help='visualization')
  args = parser.parse_args()
  main(args)
