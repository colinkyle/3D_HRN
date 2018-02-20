import cv2
import copy
import fsys
from imtools import Img3D, MRI, get_defaultMETA, load_obj, save_obj, affine_2d_to_3d, affine_make_full, affine_get_params, affine2d_from_xytheta, reverse_tform_order, deformable_reg_batch
import re
import matplotlib.pyplot as plt
import numpy as np
from TF_models import AtlasNet, ElasticNet


fsys.cd('D:/__Atlas__/data/35520/histology')

# generate edge network
if False:
  # Build neighbor distributions
  IMG = Img3D(fsys.file('*.tiff'))
  metaList = []
  pixel_size = 0.244
  section_size = .120
  for i,f in enumerate(IMG.fList):
    meta = get_defaultMETA()
    meta['fname'] = f
    # set x
    meta['MAT'][0][0] = pixel_size
    meta['MAT0'][0][0] = pixel_size
    # set y
    meta['MAT'][1][1] = pixel_size
    meta['MAT0'][1][1] = pixel_size
    # set z
    meta['MAT'][2][2] = pixel_size
    meta['MAT0'][2][2] = pixel_size

    # set x offset
    meta['MAT'][0][3] = -pixel_size * IMG.images[i].shape[0] / 2
    meta['MAT0'][0][3] = -pixel_size * IMG.images[i].shape[0] / 2

    # set y offset
    meta['MAT'][1][3] = -pixel_size * IMG.images[i].shape[1] / 2
    meta['MAT0'][1][3] = -pixel_size * IMG.images[i].shape[1] / 2

    # set z offset
    meta['MAT'][2][3] = pixel_size * (i - len(IMG.images) / 2)
    meta['MAT0'][2][3] = pixel_size * (i - len(IMG.images) / 2)

    metaList.append(meta)

  IMG.update_meta(metaList)
  IMG.build_edge_net(3)
  IMG.estimate_edge_distributions()
  save_obj(IMG,'35520_hist.obj')
  print('Done!')
  exit()
# load histology and edge network
#IMG = load_obj('TestImg3D.obj')
if False:
  IMG = load_obj('35520_hist.obj')
  metaList = []
  pixel_size = 0.244
  section_size = .120

  locs = [int(re.search('[0-9]*[N]', IMG.fList[i]).group(0)[0:-1]) for i in range(len(IMG.fList)) if
          re.search('[0-9]*[N]', IMG.fList[i])]
  z_middle = (max(locs) + min(locs)) / 2.
  for i, f in enumerate(IMG.fList):
    meta = get_defaultMETA()
    meta['fname'] = f
    # set x
    meta['MAT'][0][0] = pixel_size
    meta['MAT0'][0][0] = pixel_size
    # set y
    meta['MAT'][1][1] = pixel_size
    meta['MAT0'][1][1] = pixel_size
    # set z
    meta['MAT'][2][2] = section_size
    meta['MAT0'][2][2] = section_size

    # set x offset
    meta['MAT'][0][3] = -pixel_size * IMG.images[i].shape[0] / 2
    meta['MAT0'][0][3] = -pixel_size * IMG.images[i].shape[0] / 2

    # set y offset
    meta['MAT'][1][3] = -pixel_size * IMG.images[i].shape[1] / 2
    meta['MAT0'][1][3] = -pixel_size * IMG.images[i].shape[1] / 2

    # set z offset
    z_pos = int(re.search('[0-9]*[N]',f).group(0)[0:-1])
    z_mm = .12 * (z_pos - z_middle)
    meta['MAT'][2][3] = z_mm
    meta['MAT0'][2][3] = z_mm

    metaList.append(meta)
# load MRI
fsys.cd('D:/__Atlas__/data/35520/mri')
mri = MRI('35520_brain_last2.nii')
#mri.affine = .885*mri.affine
fsys.cd('D:/__Atlas__/data/35520/histology')
IMG = load_obj('35520_hist.obj')
# calculate histology to MRI
if True:
  fsys.cd('D:/__Atlas__')
  # call some function that:
  # takes ranges for parameters
  #   Iteratively
  #     1. Retrains network
          # gen big batch
          # sample big batch
  #     2. Samples parameter space, registers, computes cost, updates parameters

  # set limits (in mm)
  zlim = [-8,8]
  theta_xlim = [-15,15]
  theta_ylim = [-15,15]
  xy_res = [.7,1.1]
  z_res = [.7,0.9]
  # do search

  score,param = mri.param_search(IMG, zlim, theta_xlim, theta_ylim, xy_res, z_res)
  # for z in range(-15,-4):
  #   (warpList, warpParam, cost_cc) = mri.reg_to_slice_TF(IMG,dz=1,theta_x=z,theta_y=0)
  #   print(z,np.mean(cost_cc))
  #   #fsys.cd('D:/__Atlas__/Simulated')
  #   #save_obj((warpList_nn, warpCoef_nn),'warpParamNN.obj')
  #   #(warpList,warpCoef) = load_obj('warpParamNN.obj')
  #
  #   # test fit
  #   p_consistency = IMG.score_param_consistency(warpParam)
  #   print(.6*np.mean(cost_cc) + .4*p_consistency)
  print(score,param)

if False:
  z = -1.0#-0.5 #0
  theta_x = 7.5#8.0# 6.0
  theta_y = 4.5#5.0# 5.5
  dxy = 0.89375#0.903125#0.92187
  dz = 1.053125#1.11875#0.96875
  params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_257x265_507000',
            'save_file': 'regNETshallow',
            'save_interval': 1000,
            'batch_size': 32,
            'lr': .0001,  # Learning rate
            'rms_decay': 0.9,  # RMS Prop decay
            'rms_eps': 1e-8,  # RMS Prop epsilon
            'width': 265,
            'height': 257,
            'numParam': 3,
            'train': True}
  net = AtlasNet(params)
  batch = mri.gen_batch(IMG,z,theta_x,theta_y,dxy,dz)
  net = mri.retrain_TF(net, batch, ntrain=2000, nbatch=32)
  tformed, xytheta, cost_cc = net.run_batch(batch)
  for i in range(batch.shape[0]):
    batch[i,:,:,1] = np.squeeze(tformed[i,:,:])
  # load elastic net
  params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic2_257x265_1000'
  params['save_file'] = 'Elastic2'
  net2 = ElasticNet(params)
  net2 = mri.retrain_TF_E(net2, batch, ntrain=2000, nbatch=32)
  warped, theta, cost_cc, cost = net2.run_batch(batch)
  #warped,cc,E = deformable_reg_batch(batch)
  # for i in range(batch.shape[0]):
  #   f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(20, 7))
  #   ax1.imshow(np.array(batch[i, :, :, 0]))
  #   ax2.imshow(np.array(warped[i, :, :, 1]))
  #   merged = np.dstack(
  #     (np.array(batch[i, :, :, 0]), np.array(warped[i, :, :, 1]), np.array(warped[i, :, :, 1])))
  #   ax3.imshow(merged)
  #   plt.show()
  for i in range(batch.shape[0]):
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(20, 7))
    merged = np.dstack(
        (np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0])))
    plt.imshow(merged)

    plt.subplot(2, 2, 2)
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(20, 7))
    merged = np.dstack(
      (np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1])))
    plt.imshow(merged)

    #plt.title(cost_cc[i])
    plt.subplot(2, 2, 3)
    merged = np.dstack(
      (np.array(batch[i, :, :, 0]), np.array(warped[i, :, :, 0]), np.array(warped[i, :, :, 0])))
    plt.imshow(merged)
    # plt.subplot(2, 2, 3)
    # plt.imshow(dxy[i, :, :, 0])
    # plt.colorbar()
    # plt.subplot(2, 2, 4)
    # plt.imshow(dxy[i, :, :, 1])
    # plt.colorbar()
    plt.show()