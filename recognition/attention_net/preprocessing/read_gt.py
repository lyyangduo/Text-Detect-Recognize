# File to read the ground-truth for various
# scene text datasets.
# Author: Ankush Gupta
# Date: 3 Jan, 2017

from utils.etree_to_dict import etree_to_dict
import xml.etree.ElementTree as ET
import os.path as osp
import numpy as np
import glob, re
import json


def parse_ggt_format(data_dir):
  """
  Parses Google OCR json and returns the
  ground-truth in standard format:
  image-id --> [5xn, [text_1,text_2,..., text_n]] dict.
  Where 5xn is a matrix for n boxes, with first four
  numbers being x,y,w,h and the 5th being the "score".
  """
  gt_fnames = glob.glob(osp.join(data_dir,'*.json'))

  # regex's for parsing image filename:
  id_match = re.compile('.*?([\d]+_*[\d]*).*')

  gt = {}
  # process each ground-truth file:
  for gt_fn in gt_fnames:
    # get the image id:
    im_id = id_match.match(osp.basename(gt_fn)).groups()[0]
    # im_id = osp.splitext(osp.basename(gt_fn))[0] ## for nips rebuttal -- remove
    # parse the json data:
    gt_dat = []
    try:
      with open(gt_fn,'r') as f:
        gt_dat = json.load(f)
    except:
      gt[im_id] = (np.empty((5,0)),[])
    # parse the bounding box and gt-string information on each line
    bbs, txt = np.zeros((5,0)), []
    for i in xrange(1,len(gt_dat)):
      try:
        i_gt = gt_dat[i]
        if isinstance(i_gt['boundingPoly'],list):
          verts = [i_gt['boundingPoly'][i]['vertices'] for i in xrange(len(i_gt['boundingPoly']))]
        else:
          verts = i_gt['boundingPoly']['vertices']

        x1y1 = np.array([[verts[0]['x']],[verts[0]['y']]])
        x2y2 = np.array([[verts[2]['x']],[verts[2]['y']]])
        wh = x2y2 - x1y1
        try:
          score = i_gt['score']
        except KeyError:
          score = 1
        bbs = np.c_[bbs, np.r_[x1y1,wh,[[score]]]]
        txt.append(i_gt['description'])
        gt[im_id] = (bbs,txt)
      except KeyError:
        continue
  return gt


def icdar13_gt(data_dir, data_set='test'):
  """
  Given the directory containing
  {train,test}_{im,gt}, returns the
  image-name --> {bbs (5xn),text} dictionary.

  Good for BOTH ICDAR 2013 and 2011.
  """
  assert(data_set in ['train','test'])

  gt_dir = osp.join(data_dir, '%s_gt'%data_set)
  gt_fnames = glob.glob(osp.join(gt_dir,'*.txt'))
  n_gt = len(gt_fnames)

  id_match = re.compile('.*_(\d+).*')
  re_match = re.compile('(\d+).?\s?(\d+).?\s?(\d+).?\s?(\d+).?\s?"([\S ]+)"')
  gt = {} # place-holder for the output
  # process each ground-truth file:
  for gt_fn in gt_fnames:
    # get the image id:
    im_id = id_match.match(osp.basename(gt_fn)).groups()[0]
    # get the text data:
    with open(gt_fn,'r') as f:
      lines = f.readlines()
    # parse the bounding box and gt-string information on each line
    bbs, txt = np.zeros((5,0)), []
    for l in lines:
      l = re_match.match(l).groups()
      xywh = np.array([[int(x) for x in l[:4]]]).T
      xywh[2:] = xywh[2:] - xywh[:2]
      xywhs = np.r_[xywh,[[1]]]
      bbs = np.c_[bbs,xywhs]
      txt.append(l[-1])
      gt[im_id] = (bbs,txt)
  return gt


def svt_gt(data_dir, data_set='test'):
  """
  Given the directory containing
  {train,test}_{im,gt}, returns the
  image-name --> {bbs (5xn),text} dictionary.
  """
  assert(data_set in ['train','test'])

  gt_fname = osp.join(data_dir,data_set+'.xml')
  #gt_dir = osp.join(data_dir, '%s_gt'%data_set)
  # convert xml to dict:
  gt_xml = ET.parse(gt_fname).getroot()
  gts = etree_to_dict(gt_xml)['tagset']['image']
  n_gt = len(gts)

  gt = {} # place-holder for the output
  for i in xrange(n_gt):
    try:
      i_gt = gts[i]
      im_id = osp.splitext(osp.basename(i_gt['imageName']))[0]
      bbs, txt = np.zeros((5,0)), []
      gt_bbs = i_gt['taggedRectangles']['taggedRectangle']
      if not isinstance(gt_bbs,list):
        gt_bbs = [gt_bbs]
      xywh_ks = ['@x','@y','@width','@height']
      for gt_bb in gt_bbs:
        xywh = np.array([float(gt_bb[k]) for k in xywh_ks])
        xywh = np.atleast_2d(xywh).T
        xywhs = np.r_[xywh,[[1]]] # append a fake "score"
        bbs = np.c_[bbs,xywhs]
        txt.append(gt_bb['tag'])
      gt[im_id] = (bbs,txt)
    except KeyError:
      continue
  return gt


def get_gt(dataset_name,data_dir,split='test'):
  if dataset_name in ['ic13','icdar13','ic11','icdar11']:
    return icdar13_gt(data_dir,split)
  elif dataset_name in ['svt']:
    return svt_gt(data_dir,split)
  else:
    raise Exception('Unkown dataset: %s'%dataset_name)


def get_pred(data_dir):
  return parse_ggt_format(data_dir)
