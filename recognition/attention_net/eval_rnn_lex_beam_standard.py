"""
Script to evaluate the RNN-model out of tensorflow graph.
(for sanity checking..)

Author: Ankush Gupta
Date: 29 Jan, 2017
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os.path as osp
import tensorflow as tf
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cPickle as cp
import scipy.ndimage as scim

import scipy
import os

#  import the various models:
from scannet.models.AttnRNN2D import AttnRNN2D
from scannet.models.DilatedResnetRecog import DilatedResnetRecog
from scannet.utils.colorize import *
import scannet.eval.eval_rnn as eval_rnn
from scannet.data_utils import batches as bat
from scannet.data_utils.balancedsymb_data import SynthBlocksData
from scannet.utils import utils
import scannet.utils.image_utils as imu

from scannet.data_utils.synth90k_data import Synth90kData


def str2ind(s,charset):
  # +1 for TF softmax:
  ind = [charset.index(ch)+1 for ch in s.lower()]
  return ind

def preproc_image(im_name, im_out_size, rgb=False):
  bucket_edges = (40*np.arange(1,15)).tolist()
  # convert to grayscale and tile:
  if rgb:
    im = scim.imread(im_name, mode='RGB')
  else:
    im = scim.imread(im_name, mode='L')
    im = np.tile(im[:,:,None], [1,1,3])
  im = eval_rnn.preprocess_single_rgb_image(im, im_out_size, bucket_edges)
  im = np.expand_dims(im, 0) # add a fake batch dimension
  return im

def do_eval(opts):
  """
  evaluate the network on small datasets (on CPU):
  """
  # tf.reset_default_graph()
  # define the alphabet:
  charset = ' 0123456789abcdefghijklmnopqrstuvwxyz'
  alphabet_size = len(charset)
  char_mask = None
  out_im_sz = [opts.image_height,None,3]
  bucket_edges = (40*np.arange(1,15)).tolist()
  bsz = 1
  BEAM_SIZE = opts.beam

  # build the model:
  d_decoder = 1024 if opts.cell=='LSTM' else 2048
  if opts.model=='DilatedResnetRecog':
    net = DilatedResnetRecog( nalphabet=alphabet_size, cell=opts.cell,
                              d_decoder=d_decoder, h_attention=opts.h_attention,
                              d_attn=512, check=False)
  else:
    raise Exception('Unknown model: '+opts.model)

  # get the test data:
  data_dir = osp.join(opts.datadir,opts.dataset)
  gt_fname, im_dir = osp.join(data_dir, 'gt.cp'), osp.join(data_dir, 'img')
  with open(gt_fname, 'rb') as f:
    test_gt = cp.load(f)
  n_im = len(test_gt['imname'])

  # setup the beam-decoder (builds the TF graph):
  dset_lex = None
  if opts.lex_type == 'dset':
    dset_lex = test_gt['dset_lex'][opts.lex_name]
    dset_lex = [w.lower() for w in dset_lex]
  decoder_func = net.setup_beam_decoder(charset, out_im_sz[0],
                                        dset_lex=dset_lex)

  # start a new session:
  config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.80
  session = tf.Session(config=config)
  init_ops = [tf.initialize_all_variables(), tf.initialize_local_variables()]
  session.run(init_ops)
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess=session,coord=coord)

  # restore checkpoint:
  if tf.gfile.Exists(opts.netfile) or tf.gfile.Exists(opts.netfile+'.index'):
    print('RESTORING MODEL from: '+opts.netfile)
    vars_to_restore = tf.get_collection(tf.GraphKeys.VARIABLES)
    restorer = tf.train.Saver(var_list=vars_to_restore)
    restorer.restore(session,opts.netfile)
  else:
    raise Exception('model file does not exist at: '+opts.netfile)

  # loop over each image individually and get the predictions:
  i_samp, corr = 0, []
  for im_idx in xrange(n_im):
    # load and pre-process the image:
    im_name = osp.join(im_dir, test_gt['imname'][im_idx])
    gt_str = test_gt['gt'][im_idx]
    # check if the gt-str is long enough:
    if len(gt_str) < opts.min_gt_len or not gt_str.isalnum():
      try:
        print(blue('skipping: %s'%gt_str))
      except:
        print('skipping: %s'%gt_str)
      continue
    im = preproc_image(im_name, out_im_sz, rgb=opts.rgb)

    # [optionally] get the word-level lexicon:
    sample_lex = None
    if opts.lex_type == 'sample':
      sample_lex = test_gt['sample_lex'][opts.lex_name][im_idx]
      sample_lex = [w.lower() for w in sample_lex]

    # run through the model:
    pred_str, pred_score = decoder_func(session, im,
                            beam_size=BEAM_SIZE, sample_lex=sample_lex)
    corr.append(utils.same_words(pred_str.lower(),gt_str.lower()))
    # print stats:
    print('[%d]'%i_samp +
          green(' gt: ') + '%-50s'%gt_str +
          green(' pred: ') + '%-50s'%pred_str +
          green(' accuracy: ') + '%.4f'%(np.sum(corr)/(len(corr)+0.0)))
    i_samp += 1
  print(green('Final result: Total correct: %d, ' +
              'Total Valid GT samples: %d')%(np.sum(corr),len(corr)))

  # save accuracy stats:
  if opts.statfile is not None:
    if osp.exists(opts.statfile):
      with open(opts.statfile,'rb') as f:
        d = cp.load(f)
    else:
      d = {}
    d[opts.netfile] = np.sum(corr)/(len(corr)+0.0)
    with open(opts.statfile,'wb') as f:
      cp.dump(d,f)

  return corr


if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Evaluate RNN model')
  parser.add_argument('model',choices=['DilatedResnetRecog'], help='model type.')
  parser.add_argument('dataset', choices=['IC03', 'IC13', 'IC15', 'SVT', 'IIIT5K'])
  parser.add_argument('netfile',help='network file to load.')
  parser.add_argument('datadir',help='directory containing the dataset.')
  parser.add_argument('image_height',type=int,help='height of the image.')
  parser.add_argument('--cell',choices=['LSTM','GRU'],default='LSTM',help='Decoder RNN cell.')
  parser.add_argument('--h-attention',action='store_true',help='Use H (of LSTM) for attention, instead of C.')
  parser.add_argument('--min-gt-len',type=int,default=3,help='Minimum length of the gt word to include in evaluation.')
  parser.add_argument('--statfile',type=str,default=None,help='file in which to store results.')
  parser.add_argument('--rgb',action='store_true',help='If specified RGB channels are used, else B/W.')
  parser.add_argument('--lex-type',type=str,default=None, choices=['sample', 'dset'], help='Lexicon type: per-sample / dataset-level.')
  parser.add_argument('--lex-name',type=str,default=None, help='Name of the lexicon.')
  parser.add_argument('--beam',type=int,default=1, help='Beam size.')

  args = parser.parse_args()
  do_eval(args)

