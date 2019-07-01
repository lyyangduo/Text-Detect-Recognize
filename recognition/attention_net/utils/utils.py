import os
import sys
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import editdistance
import numpy as np 


class strLabelConverter(object):
    """Convert between str and label.
    Args:
        alphabet (str): set of the possible characters.
    """

    def __init__(self, alphabet, ignore_case=True):

        alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i 

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            list of ind
        """
        if isinstance(text, str):
            text = [self.dict[char.lower()] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return text, length

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str)
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 :
                        char_list.append(self.alphabet[t[i]])
                    else:
                        break
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

def setup_logger(log_file_path):
    """
    adapted from https://github.com/lvpengyuan/ssd.tf/blob/fctd-box/src/utils.py
    Setup a logger that simultaneously output to a file and stdout
    ARGS
    log_file_path: string, path to the logging file
    """
    # logging settings
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # file handler
    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    # stream handler (stdout)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_file_path)

def print_args(args):
    for arg in vars(args):
        logging.info(arg + ':%s'%(getattr(args, arg)))


def lex_free_acc(preds,labels,converter):
    """
    preds: output from the network, [B,max_len,nClass]
    labels: groundtruth. list of index in length [B*max_len]

    output: acc and two strings
    """
    n_correct = 0
    preds = torch.argmax(preds,dim =2)
    preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
    ## decode
    preds = preds.contiguous().view(-1)
    text_preds = converter.decode(preds.data, preds_size, raw=False)
    text_labels = converter.decode(labels, preds_size, raw=False)
    if isinstance(text_preds, str):
        text_preds = [text_preds]
        text_labels = [text_labels]

    for pred, target in zip(text_preds,text_labels):
        if pred == target:
            n_correct += 1

    acc = n_correct*1.0/len(text_preds)
    return acc,pred,target


def lex_acc(args,lexicon,preds,gt_str,converter):
    preds_ = torch.argmax(preds,dim =2)

    preds_size = torch.IntTensor([preds_.size(1)] * preds_.size(0))
    preds_ = preds_.contiguous().view(-1)

    pred_str = converter.decode(preds_.data,preds_size, raw=False)
    words,inds,probs,pred = [],[],[],[] 
    n_correct = 0

    for ind,lex in enumerate(lexicon):

        lex = lex.lower()
        dis = editdistance.eval(lex,pred_str)
        if dis < args.max_ed:
            words.append(lex)
            inds.append(ind)
    for i in range(len(inds)):
        lex_enc = converter.encode(words[i])
        prob =1.0
        for k1,k2 in enumerate(lex_enc):
            prob = prob*preds[:,k1,int(k2[0])]
        probs.append(prob)
    if probs != []:
        pred = words[probs.index(np.max(probs))]


    acc = n_correct*1.0/len(pred)
    return acc,pred_str



def mask(args,labels,device):

    masks = torch.zeros([args.batch_size,args.max_len]).to(device)
    for n,label in enumerate(labels):
        length = torch.nonzero(label).shape[0]+1 
        masks[n,:length] = torch.ones([length])
    return masks
