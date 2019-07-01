"""
Utility functions for bounding-boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.spatial as ssp
import scipy.spatial.distance as ssd
from unionfind import unionfind

import shapely.geometry as sgm



def find_homography(x1,x2):
  """
  Given two 2D point-sets (Nx2), x1 and x2, returns the 3x3 homography matrix H, s.t.:
    x2 = H*x1
  """
  n_pts = x1.shape[0]
  x1_h = np.c_[x1, np.ones((n_pts,1))]
  # get the coorindate matrix:
  Ax = np.c_[ -x1_h,  0*x1_h,  x1_h*x2[:,0][:,None]]
  Ay = np.c_[0*x1_h,   -x1_h,  x1_h*x2[:,1][:,None]]
  A = np.r_[Ax,Ay]
  # get the vector associated with the smallest singular value:
  u,s,v = np.linalg.svd(A)
  H = v[-1,:] # last row of V are elements of H
  H = np.reshape(H,[3,3])
  return H


def quad_transform_bb(imsize,quad,bb):
  """
  Applies the PIL's Quad to Rectangle transform to xy-coordinates:
  Assumes that the QUAD parameters are as specified for the PIL.transform function.
  """
  quad = np.reshape(quad,[4,2])
  w,h = imsize
  rect = np.array([[0,0],[0,h],[w,h],[w,0]],dtype=np.float32)
  H = find_homography(quad,rect)
  # apply homography to BB points:
  bb = np.reshape(bb,[2,-1])
  bb = np.r_[bb,np.ones((1,bb.shape[-1]))]
  bbo = H.dot(bb)
  bbo[:2,:] = bbo[:2,:] / bbo[2,:][None,:]
  bbo = np.reshape(bbo[:2,:],[2,4,-1])
  return bbo


def plot_rects(rects,im=None,color='r'):
  """
  Plots 2x4xN rectangles.
  """
  if isinstance(rects,np.ndarray):
    rects = [rects[:,:,i] for i in xrange(rects.shape[-1])]
  # visualize the lines:
  if im is not None:
    plt.imshow(im)
  plt.hold(True)
  for r in rects:
    rr = np.c_[r,r[:,0]]
    plt.plot(rr[0,:],rr[1,:],color)
  plt.hold(False)


def arrange_bb(bb):
  """
  Given bounding-box coords: 2x4xN,
  arranges them in CW order, with the
  first point being the one closest
  to the origin.
  """
  dists = np.linalg.norm(bb,axis=0)
  top_left = np.argsort(dists)[0]
  bb = np.c_[bb[:,top_left:],bb[:,:top_left]]
  # ensure CCW:
  pp = sgm.Polygon(bb.T)
  pp = sgm.polygon.orient(pp)
  bb_ccw = np.array([pt for pt in pp.exterior.coords])[:-1,:].T
  return bb_ccw



def bb_vitals(bb):
  """
  BB: 2x4 array of coordinates.

  Assumes:
      [x1,y1,x2,y2,....,x4,y4] are clockwise points, starting from
      the top-left coordinate.
  """
  x_ax = 0.5 * (bb[:,1]-bb[:,0] + bb[:,2]-bb[:,3])
  w = np.linalg.norm(x_ax)
  h = np.mean([np.linalg.norm(bb[:,3]-bb[:,0]),np.linalg.norm(bb[:,2]-bb[:,1])])
  x_u = x_ax / (w+1e-8)
  y_u = np.array([-x_u[1], x_u[0]])
  o = bb[:,0]
  return o,h,w,x_u,y_u


def xywh2coords(bb):
  """
  BB: Nx4 tensor (X,Y,W,H).
  Returns: 2x4xN coordinates.
  """
  bb = np.atleast_2d(bb)
  x,y,w,h = bb[:,0],bb[:,1],bb[:,2],bb[:,3]
  coords = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
  coords = np.transpose(coords, [1,0,2])
  return coords


def pad_rect(xy,h,f_pad):
  """
  PAD boxes by F_PAD*H amount.
  XY: 2x4xN matrix.
  """
  pad = np.array([[-1,-1],[1,-1],[1,1],[-1,1]]).T
  pad = pad * f_pad * np.array([h,h])[:,None]
  xy = xy + np.expand_dims(pad, axis=2)
  return xy


def point_axis_distance(o,xax,yax,p,x_std,y_std):
  """
  Returns the Mahalanobis distance,
  b/w points O and P, with the x-axis along XAX (y along YAX),
  with the standard deviations being X_STD,Y_STD respectively.
  """
  R = np.diag([1.0/x_std,1.0/y_std]).dot(np.c_[xax,yax].T)
  t = R.dot(p-o)
  return np.sqrt(t.dot(t))


def find_rotated_rect(pts):
  """
  Find a rotated rectangle with minimum width,
  given a set of N points (PTS: Nx2 array).

  Implements the rotated callipers algorithm.
  """
  def get_rect(eq,v_p,verts):
    """
    Returns the height and width of the
    rotated rectangle.
    """
    n_pts = verts.shape[0]
    v,b = eq[:2],eq[2]
    v = v / np.linalg.norm(v)
    v_p = v_p / np.linalg.norm(v_p)
    perp,proj = [],[]
    for i in xrange(n_pts):
      p = verts[i,:]
      d_p = v.dot(p) + b
      perp.append(d_p)
      proj.append(v_p.dot(p - d_p * v))
    h = np.min(perp)
    w = np.max(proj)-np.min(proj)
    # get the rectangle:
    m_pt = -b*v + v_p * np.min(proj)
    rect = [m_pt + h*v,
            m_pt + h*v + w*v_p,
            m_pt + w*v_p,
            m_pt]
    rect = np.array(rect)
    return np.abs(h)*np.abs(w),rect
  # Rotating callipers algorithm:
  cvx_hull = ssp.ConvexHull(pts)
  eq,simplices = cvx_hull.equations,cvx_hull.simplices
  verts = pts[cvx_hull.vertices,:]
  n_sides = eq.shape[0]
  areas,rects = [],[]
  for i in range(n_sides):
    simplex = simplices[i,:]
    w_p = pts[simplex[1],:]-pts[simplex[0],:]
    a,rect = get_rect(eq[i,:],w_p,verts)
    areas.append(a)
    rects.append(rect)
  # select the rect with the minimum area:
  rect_idx = np.argmin(areas)
  rot_rect = arrange_bb(rects[rect_idx].T)
  return rot_rect


def find_lines_poly(word_bb,im=None):
  """
  Given the 2x4XN matrix of BB coordinates,
  merges the boxes into lines.
  """
  if np.size(word_bb)==0:
    return [],[]

  edge_thresh = 3.0
  height_factor = 0.50
  def f_dist(bb1,bb2):
    """
    L2 distance b/w the center-point
    on the right-edge of BB1 to the
    center-point on the left-edge of BB2.
    """
    bb1 = np.reshape(bb1,[2,4])
    bb2 = np.reshape(bb2,[2,4])
    o1,h1,w1,x1,y1 = bb_vitals(bb1)
    o2,h2,w2,x2,y2 = bb_vitals(bb2)
    min_h = min(h1,h2)
    # distance b/w the right-edge and the left-edge:
    p_bb1 = o1 + x1*w1 + y1*h1/2.0
    p_bb2 = o2 + y2*h2/2.0
    d = point_axis_distance(p_bb1,x1,y1,p_bb2,edge_thresh*min_h,height_factor*min_h)
    edge_close = (d <= 1.0)
    return np.float(edge_close)

  # get all-pairs distances b/w the boxes:
  word_bb_flat = np.reshape(word_bb,[8,-1])
  dists = ssd.cdist(word_bb_flat.T,word_bb_flat.T,f_dist)
  # create groups:
  n_bb = word_bb.shape[-1]
  U = unionfind(n_bb)
  for i in xrange(n_bb):
    for j in xrange(n_bb):
      if i==j: continue
      if dists[i,j] > 0:
        U.unite(i,j)
  # get the lines:
  lines = U.groups()
  # get the combined coordinates:
  line_rects = []
  for l in lines:
    line_box_pts = word_bb[:,:,l]
    line_box_pts = np.reshape(line_box_pts,[2,-1]).T
    line_rects.append(find_rotated_rect(line_box_pts))
  line_rects = np.transpose(np.array(line_rects),[1,2,0])
  return line_rects,lines


def find_blocks_poly(line_bb,line_word_inds,im=None):
  """
  Given a tensor of 2x4xN rotated rectangles of lines,
  groups the lines together that "touch".

  LINE_WORD_INDS: indices of the words, which form the lines.
  """
  if np.size(line_bb)==0:
    return [],[],[]

  def extend_bb(bb,up=True,f_h=2.0):
    """
    extend the bounding-box, making the height bigger:
    """
    o,h,w,x,y = bb_vitals(bb)
    c = np.mean(bb,axis=1)
    if up:
      f_u,f_d = f_h,0.5
    else:
      f_u,f_d = 0.5,f_h
    v = [c-w/2*x - f_u*h*y,
         c+w/2*x - f_u*h*y,
         c+w/2*x + f_d*h*y,
         c-w/2*x + f_d*h*y]
    return np.array(v).T

  def should_merge(bb1,bb2,area_intersection_thresh=0.30):
    """
    Given two line BBs, determine,
    if they should be merged or not.
    """
    c1,c2 = np.mean(bb1,axis=1), np.mean(bb2,axis=1)
    # make BB1 to be at the "top":
    if c1[1] > c2[1]:
      t = bb1
      bb1 = bb2
      bb2 = t
    # extend the boxes in the appropriate directions:
    bb1_ex,bb2_ex = extend_bb(bb1,up=False), extend_bb(bb2,up=True)
    bb1_p,bb2_p = sgm.Polygon(bb1_ex.T), sgm.Polygon(bb2_ex.T)
    # get the maximum area of the line:
    max_a = max(sgm.Polygon(bb1.T).area, sgm.Polygon(bb2.T).area)
    # get the area of intersection:
    int_a = bb1_p.intersection(bb2_p).area
    return np.float(int_a / (max_a + 0.0) >= area_intersection_thresh)

  n_bb = line_bb.shape[-1]
  U = unionfind(n_bb)
  for i in xrange(n_bb):
    for j in xrange(n_bb):
      if i==j: continue
      if should_merge(line_bb[:,:,i],line_bb[:,:,j],area_intersection_thresh=0.7):
        U.unite(i,j)
  # get the lines:
  blocks = U.groups()
  # get the combined coordinates:
  block_rects = []
  block_word_inds = []
  block_line_inds = []
  for b in blocks:
    block_pts = line_bb[:,:,b]
    block_pts = np.reshape(block_pts,[2,-1]).T
    block_rects.append(find_rotated_rect(block_pts))
    block_line_inds.append(b)
    b_ind = []
    for ib in b:
      b_ind += line_word_inds[ib]
    block_word_inds.append(b_ind)
  block_rects = np.transpose(np.array(block_rects),[1,2,0])
  return block_rects,block_word_inds,block_line_inds
