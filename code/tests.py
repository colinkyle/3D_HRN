import cv2
import copy
import fsys
from imtools import Img3D, MRI, get_defaultMETA, load_obj, save_obj, affine_2d_to_3d, affine_make_full, affine_get_params
import matplotlib.pyplot as plt
import numpy as np
from TF_models import AtlasNet


fsys.cd('/Users/colin/Dropbox/__Atlas__/Simulated')

# generate edge network
if False:
  # Build neighbor distributions
  IMG = Img3D(fsys.file('*.tiff'))
  metaList = []
  for i,f in enumerate(IMG.fList):
    meta = get_defaultMETA()
    meta['fname'] = f
    # set x
    meta['MAT'][0][0] = 0.737
    meta['MAT0'][0][0] = 0.737
    # set y
    meta['MAT'][1][1] = 0.737
    meta['MAT0'][1][1] = 0.737
    # set z
    meta['MAT'][2][2] = 0.737
    meta['MAT0'][2][2] = 0.737

    # set x offset
    meta['MAT'][0][3] = -0.737 * IMG.images[i].shape[0] / 2
    meta['MAT0'][0][3] = -0.737 * IMG.images[i].shape[0] / 2

    # set y offset
    meta['MAT'][1][3] = -0.737 * IMG.images[i].shape[1] / 2
    meta['MAT0'][1][3] = -0.737 * IMG.images[i].shape[1] / 2

    # set z offset
    meta['MAT'][2][3] = 0.737 * (i - len(IMG.images) / 2)
    meta['MAT0'][2][3] = 0.737 * (i - len(IMG.images) / 2)

    metaList.append(meta)

  IMG.update_meta(metaList)
  IMG.build_edge_net(3)
  IMG.estimate_edge_distributions()
  print('Done!')
# load histology and edge network
IMG = load_obj('TestImg3D.obj')
# load MRI
mri = MRI('mri_small.nii')

# calculate histology to MRI
if True:
  fsys.cd('/Users/colin/Dropbox/__Atlas__/')
  (warpList_nn, warpCoef_nn) = mri.reg_to_slice_TF(IMG,dz=20,theta_x=-10,theta_y=-1)
  (warpList_cv, warpCoef_cv) = mri.reg_to_slice(IMG, dz=20, theta_x=-10, theta_y=-1)

fsys.cd('/Users/colin/Dropbox/__Atlas__/Simulated')
save_obj((warpList_nn, warpCoef_nn),'warpParamNN.obj')
save_obj((warpList_cv, warpCoef_cv),'warpParamCV.obj')
# Load stored hist -> mri params
#(warpList,warpCoef) = load_obj('warpParam4.obj')
# for index in range(len(warpList)):
#   img = IMG.images[i]
#   cc = warpCoef[i]
#   warp = warpList[i]

# 0 -> mri -> 1
# Err = []
# total_err = 0
# for pair in IMG.edges.keys():
#   warp2 = np.dot(np.linalg.inv(affine_2d_to_3d(warpList[pair[0]])),affine_2d_to_3d(warpList[pair[1]]))
#   _, _, mri_tz, mri_x, mri_y, _ = affine_get_params(warp2)
#   edge_x, edge_y, edge_tz = IMG.edges[pair].param_x[1],IMG.edges[pair].param_y[1], -IMG.edges[pair].param_theta[0]
#
#   p_x = IMG.edges[pair].p_x(mri_x)
#   p_y = IMG.edges[pair].p_y(mri_y)
#   p_theta = IMG.edges[pair].p_theta(mri_tz)
#
#   p_edge = p_x*p_y*p_theta*(1-.05**np.diff(pair))
#
#   if not np.isnan(p_edge):
#     total_err += np.log(p_edge)
#
#   Err.append(((mri_x-edge_x) ** 2) + ((mri_y-edge_y) ** 2) + ((mri_tz-edge_tz) ** 2))
#   print(pair,Err[-1])
#
# print(total_err)
# print(np.exp(total_err))
# print('Loaded')
