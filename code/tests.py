import copy
import matplotlib.pyplot as plt
import numpy as np
import re
import nibabel as nib
from scipy.misc import imsave
import time

import fsys
from imtools import Stack, Template, get_defaultMETA, load_obj, save_obj
from search import objectiveFunction, nelderMeadSimplex, bfgs, bhopping, powell, iterativePowellWithRetrain
from TF_models import RigidNet, TpsNet

fsys.cd('D:/__Atlas__/data/30890_/histology/segmented')

rigid_params = {'load_file': 'D:/__Atlas__/model_saves/model-Rigid_265x257',
			  'save_file': 'Rigid',
			  'save_interval': 1000,
			  'batch_size': 32,
			  'lr': .0001,  # Learning rate
			  'rms_decay': 0.9,  # RMS Prop decay
			  'rms_eps': 1e-8,  # RMS Prop epsilon
			  'width': 265,
			  'height': 257,
			  'numParam': 3,
			  'train': True}
elastic_params = copy.deepcopy(rigid_params)
elastic_params['load_file'] = 'D:/__Atlas__/model_saves/model-TPS_265x257'
elastic_params['save_file'] = 'TPS'

# GENERATE STACK WITH EDGE DISTRIBUTIONS
if True:
	print('Creating image stack...')
	t0 = time.time()
	# histology section files
	fList = fsys.file('*.png')

	# Generate metadata list with each section's metadata.
	# Each section has a file name and 3D affine matrix:
	# for resolution and z_location.  STN code does not
	# consider each image's affine matrix.  Rigid STN
	# rotates about image center.  affine matrixes are
	# used to resample the Template to matching resolution
	# at the precise z_location to match each section.
	metaList = []
	pixel_size = 0.244
	section_size = .12 # z spacing
	#file names contain section locations (this parses)
	locs = [int(re.search('[0-9]*[N]', fList[i]).group(0)[0:-1]) for i in range(len(fList)) if
			re.search('[0-9]*[N]', fList[i])]
	z_middle = (max(locs) + min(locs)) / 2.
	for i, f in enumerate(fList):
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
		meta['MAT'][0][3] = -pixel_size * 265 / 2
		meta['MAT0'][0][3] = -pixel_size * 257 / 2

		# set y offset
		meta['MAT'][1][3] = -pixel_size * 265 / 2
		meta['MAT0'][1][3] = -pixel_size * 257 / 2

		# set z offset
		z_pos = int(re.search('[0-9]*[N]', f).group(0)[0:-1])
		z_mm = section_size * (z_pos - z_middle)
		meta['MAT'][2][3] = z_mm
		meta['MAT0'][2][3] = z_mm

		metaList.append(meta)

	stack = Stack(fList, metaList)
	print('Estimating edge distributions...')
	stack.estimate_edge_distributions(rigid_params)
	save_obj(stack, '30890_hist.obj')
	print('Done!')
	t1 = time.time()
	print('Elapsed time: {} seconds'.format(t1-t0))

# load template
print('Loading template and stack...')
fsys.cd('D:/__Atlas__/data/30890_/mri')
template = Template('_30890_.nii')
# fsys.cd('D:/__Atlas__/data/NMT_Template')
# template = Template('NMT_Template_Repositioned.nii')

fsys.cd('D:/__Atlas__/data/30890_/histology/segmented')
stack = load_obj('30890_hist.obj')

# POWELL SEARCH RECONSTRUCTION PARAMS
if True:
	t0 = time.time()
	#test new search
	netR = RigidNet(rigid_params)
	netE = TpsNet(elastic_params)

	print('starting search...')
	vals, netR, netE = iterativePowellWithRetrain(stack,template,netR,netE,[5,-5,-5,0.95],niter = 10)
	print('finished search...')

	t1 = time.time()
	print('Elapsed time: {} seconds'.format(t1 - t0))

# GRID SEARCH RECONSTRUCTION PARAMS
if False:
	t0 = time.time()
	fsys.cd('D:/__Atlas__')
	# call some function that:
	# takes ranges for parameters
	#   Iteratively
	#     1. Retrains network
	# gen big batch
	# sample big batch
	#     2. Samples parameter space, registers, computes cost, updates parameters

	# set limits (in mm)
	zlim = [-5, 5]
	theta_xlim = [-10, 10]
	theta_ylim = [-10, 10]
	xy_res = [0.85, 0.95]
	# z_res = [.7,1.1]
	# do search

	# score,param,param_dist = mri.param_search(IMG, zlim, theta_xlim, theta_ylim, xy_res, z_res)
	score, param, param_dist = template.param_search(stack, zlim, theta_xlim, theta_ylim, xy_res)
	save_obj(param_dist, 'parameter_distribution.obj')
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

	print(score, param)
	t1 = time.time()
	print(t1-t0)

# VIEW RESULTS
if False:
	z = 1.65  # -1.5#-0.5 #0
	theta_x = -6  # 3.75#7.5#8.0# 6.0
	theta_y = -0.625  # 5.5#5.625#4.5#5.0# 5.5
	dxy = 0.9125  # 0.9#0.89375#0.903125#0.92187
	dz = 0.9125# 0.925#1.053125#1.11875#0.96875
	#resample MRI
	new_mri = template.get_resampled_mri(stack, z, theta_x, theta_y, dxy, dz)
	fsys.cd('D:/__Atlas__/data/30890_/mri')
	nib.save(new_mri, 'mri_resampled.nii.gz')


	params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_265x257_512000',
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
	batch = template.gen_batch(stack, z, theta_x, theta_y, dxy, dz)

	net = template.retrain_TF(net, batch, ntrain=2000, nbatch=32)
	net.save_ckpt('D:/__Atlas__/model_saves/model-' + 'Rigid30890')

	tformed, xytheta, cost_cc = net.run_batch(batch)
	for i in range(batch.shape[0]):
		batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])
	# save out images
	# fsys.mkcd('D:/__Atlas__/data/30890_/raw_images')
	# for i in range(batch.shape[0]):
	# 	name_mri = 'mri'+str(i).zfill(2) +'.png'
	# 	name_hist = 'hist' + str(i).zfill(2) + '.png'
	# 	imsave(name_mri,batch[i,:,:,0])
	# 	imsave(name_hist,batch[i,:,:,1])
	# exit()
	# load elastic net
	params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic2_265x257_176000'
	params['save_file'] = 'Elastic2'
	net2 = ElasticNet(params)
	# V2 three-way registration
	# get batch for 2nd to -2nd -1 back
	# batch_neg1 = copy.deepcopy(batch[1:-2])
	# for i in range(batch_neg1.shape[0]):#fixed: mri -> hist(-1)
	# 	batch_neg1[i,:,:,0] = batch[i,:,:,1]
	#
	# batch_pos1 = copy.deepcopy(batch[1:-2])
	# for i in range(batch_pos1.shape[0]):#fixed: mri -> hist(+1)
	# 	batch_pos1[i,:,:,0] = batch[i+2,:,:,1]
	#
	# net2 = mri.retrain_TF_E(net2, batch_neg1, ntrain=2000, nbatch=32)
	# warped, theta_neg1, cost_cc, cost, cost2 = net2.run_batch(batch_neg1)
	#
	# net2 = mri.retrain_TF_E(net2, batch_pos1, ntrain=2000, nbatch=32)
	# warped, theta_pos1, cost_cc, cost, cost2 = net2.run_batch(batch_pos1)

	net2 = template.retrain_TF_E(net2, batch, ntrain=2000, nbatch=32)
	net2.save_ckpt('D:/__Atlas__/model_saves/model-' + 'Elastic30890')
	warped, theta_mri, cost_cc, cost, cost2 = net2.run_batch(batch)

	# get weights and compute weighted average
	# splines = copy.deepcopy(theta_mri)
	# splines[1:-2] = 0.3333334 * splines[1:-2] + 0.3333333 * theta_neg1 + 0.3333333 * theta_pos1
	# warped, splines2, cost_cc = net2.transform_batch(batch,splines)

	stack.set_warped_from_tformed(warped)
	hist_mri = stack.get_mri_from_warped()
	fsys.cd('D:/__Atlas__/data/30890_/mri')
	nib.save(new_mri, 'mri_resampled.nii.gz')
	nib.save(hist_mri, 'hist_warped_resampled_lowReg.nii.gz')
	# warped,cc,E = deformable_reg_batch(batch)
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
		# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(20, 7))
		merged = np.dstack(
			(np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0])))
		plt.imshow(merged)

		plt.subplot(2, 2, 2)
		# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(20, 7))
		merged = np.dstack(
			(np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1])))
		plt.imshow(merged)

		# plt.title(cost_cc[i])

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
