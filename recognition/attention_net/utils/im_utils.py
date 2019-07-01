from utils import bb_utils as bbu
from PIL import Image
import numpy as np

def extract_patch(im,bb,pad=None):
  """
  Return {IM_PATCH,GT_TXT,BB}.
  """
  # pad the block-box:
  o,h,w,xax,yax = bbu.bb_vitals(bb)
  if pad is not None:
    bb = bbu.pad_rect(bb,h,pad)
  # clip the coordinates of the block:
  imsz = im.size
  bb[bb < 0] = 0
  bb[0,bb[0,:]>=imsz[0]] = imsz[0]
  bb[1,bb[1,:]>=imsz[1]] = imsz[1]
  bb = np.round(bb).astype(int)
  bb_patch = bb.copy()
  # extract the patch:
  quad = tuple(bb[:,[0,3,2,1]].T.flatten())
  bbsz = (int(np.ceil(w)),int(np.ceil(h)))
  im_patch = im.transform(bbsz,Image.QUAD,quad,Image.BILINEAR)
  return im_patch
