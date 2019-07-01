from dataloader.SynthLoader import SynthLoader,text_collate
from utils.utils import setup_logger, print_args, strLabelConverter,lex_free_acc
import argparse
import _pickle as cp
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='AN')
    parser.add_argument('--name', default='bn_smaller_batch', type=str)
    
    ## data setting 
    parser.add_argument('--root', default='/scratch/local/ssd/datasets',type=str)
    parser.add_argument('--train_dataset', default='synthtext', type=str)
    parser.add_argument('--vis_gt', default=False, type=bool)
    parser.add_argument('--vis_gt_path', default='/users/czhang/data/vis', type=str)
    parser.add_argument('--load_width', default=256, type=int)
    parser.add_argument('--load_height', default=32, type=int)
    parser.add_argument('--alphabet', default=' 0123456789abcdefghijklmnopqrstuvwxyz', type=str)
    args = parser.parse_args()

    cp_dir = '/users/czhang/data/FAN/'
    converter = strLabelConverter(args.alphabet)

    trainset =  SynthLoader(args, args.train_dataset, converter)
    labels = trainset.image_labels
    multipler_dict = {1:382, 2:34, 3: 9,
         4:3,  5:2,  6:1,  7:1,
         8:1,  9:1,  10: 1, 11: 2,
         12:3, 13: 6, 14: 8}
    # lengths = []

    for ind,label in enumerate(labels):
        lengths.append(len(label))
    #     if ind % 100000 ==0:
    #         percent = int(ind*100/len(labels))
    #         print('%d percent of the data done' %percent)

    # with open(osp.join(cp_dir, 'word_lengths.cp'), 'wb') as f:
    #   cp.dump(lengths, f)
    # lengths = cp.load(open(osp.join(cp_dir, 'word_lengths.cp'),'rb'))
    # plt.figure()

    # plt.hist(lengths, bins=range(16))  
    # import ipdb;ipdb.set_trace()
    # plt.title("Histogram of word lengths")
    # plt.savefig(osp.join(cp_dir,'word_length_hist.png'))





if __name__ == '__main__':
    main()