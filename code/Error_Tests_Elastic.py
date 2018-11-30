import fsys
from imtools import Img, Edge, reverse_tform_order, affine2d_from_xytheta, ecc
import copy
import numpy as np
from TF_models import AtlasNet, ElasticNet, TrainTForm
import matplotlib.pyplot as plt
import imutils
import math
import cv2
import time
import matplotlib as mpl
import SimpleITK as sitk

mpl.rcParams['pdf.fonttype'] = 42


def calc_inverse_xform(xytheta):
	xytheta_out = copy.deepcopy(xytheta)
	for i in range(xytheta.shape[0]):
		rot = affine2d_from_xytheta([xytheta[i, 0], xytheta[i, 1], -xytheta[i, 2]])[0:2]
		pos = np.array([xytheta[i, 0], xytheta[i, 1], 1])
		dxy2 = pos[0:2] - np.sum(rot * pos, axis=1)
		xytheta_out[i] = [dxy2[0], dxy2[1], -xytheta[i, 2]]
	return xytheta_out


fsys.cd('D:/__Atlas__')

params = {'load_file': 'D:/__Atlas__/model_saves/model-Elastic2_265x257_176000',
		  'save_file': 'Elastic2',
		  'save_interval': 1000,
		  'batch_size': 32,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.999,  # RMS Prop decay
		  'rms_eps': 1e-9,  # RMS Prop epsilon
		  'width': 265,
		  'height': 257,
		  'numParam': 3,
		  'train': True}
netE = ElasticNet(params)

# retrain

if 0:
	files = fsys.file('D:/__Atlas__/data/07119/histology/segmented/*.png')
	pairs = fsys.pair_offset(files, files, offset=2)
	batch = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2), dtype=np.float32)
	for b, f in enumerate(pairs[0:32]):
		fixed = Img.imread(f[0])  # .p_intensity
		moving = Img.imread(f[1])  # .p_intensity

		matcher = sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(512)
		matcher.SetNumberOfMatchPoints(30)
		moving = matcher.Execute(sitk.GetImageFromArray(moving, sitk.sitkFloat32),
								 sitk.GetImageFromArray(fixed, sitk.sitkFloat32))
		moving = sitk.GetArrayFromImage(moving)
		batch[b, :, :, :] = np.stack((fixed, moving), axis=2)
	for _ in range(1000):
		cnt, cost, cost_t, _, _, _ = netE.train(batch)
		print('count: {}, cost: {}, cost_t: {}'.format(cnt, cost, cost_t))
		if (params['save_file']):
			if cnt % params['save_interval'] == 0:
				netE.save_ckpt('model_saves/model-' + params['save_file'] + "_" + str(params['width']) + 'x' + str(
					params['height']) + '_' + str(cnt))
				print('Model saved')

if 1:
	files = fsys.file('D:/__Atlas__/data/35717/histology/segmented/*.png')
	pairs = fsys.pair_offset(files, files, offset=2)
	batch = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2), dtype=np.float32)

	# fig,axes = plt.subplots(3,3,figsize = (15,15))
	# for s in range(3):
	Similarity_OG = np.zeros((32,))

	Similarity_TF = np.zeros((32,))

	Similarity_SE = np.zeros((32,))

	Time_SE = np.zeros((32,))

	cnt = 0
	elapsed_time_se = 0
	elapsed_time_tf = 0
	batch = np.zeros((32, 265, 257, 2))

	for im in np.arange(0, 32):
		fixed = Img.imread(pairs[im][0])
		moving = Img.imread(pairs[im][1])
		matcher = sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(512)
		matcher.SetNumberOfMatchPoints(30)
		moving = matcher.Execute(sitk.GetImageFromArray(moving, sitk.sitkFloat32),
								 sitk.GetImageFromArray(fixed, sitk.sitkFloat32))
		moving = sitk.GetArrayFromImage(moving)

		Similarity_OG[im] = ecc(fixed, moving)
		batch[im, :, :, :] = np.stack((fixed, moving), axis=2)
		# simple Elastix registration
		elastixImageFilter = sitk.ElastixImageFilter()
		elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed))
		elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))

		parameterMap = sitk.GetDefaultParameterMap("bspline")
		parameterMap['FinalGridSpacingInPhysicalUnits'] = ['32']
		# parameterMap['MaximumNumberOfIterations'] = ['1']
		elastixImageFilter.SetParameterMap(parameterMap)
		start_time = time.time()
		elastixImageFilter.Execute()
		end_time = time.time()
		Time_SE[im] = end_time - start_time
		elapsed_time_se += Time_SE[im]
		moved = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
		Similarity_SE[im] = ecc(fixed, moved)

	start_time = time.time()
	tformed, theta, cost_cc, cost, cost2 = netE.run_batch(batch)
	end_time = time.time()
	elapsed_time_tf += end_time - start_time
	for i in range(tformed.shape[0]):
		Similarity_TF[i] = ecc(batch[i, :, :, 0], tformed[i])

	print(elapsed_time_se)
	print(elapsed_time_tf)

	mc = [88. / 255, 184. / 255, 249. / 255]
	fig, ax = plt.subplots(3, 1, figsize=(10, 15))
	ax[0].scatter(Similarity_OG.flatten(), Similarity_SE.flatten(), c=mc, alpha=0.6)
	ax[0].set_xlabel('Starting CC')
	ax[0].set_ylabel('Simple Elastix CC')
	ax[0].plot([np.min([np.min(Similarity_OG), np.min(Similarity_SE)]), 1],
			   [np.min([np.min(Similarity_OG), np.min(Similarity_SE)]), 1], 'k-', dashes=[4, 2])

	ax[1].scatter(Similarity_OG.flatten(), Similarity_TF.flatten(), c=mc, alpha=0.6)
	ax[1].set_xlabel('Starting CC')
	ax[1].set_ylabel('STN CC')
	ax[1].plot([np.min([np.min(Similarity_OG), np.min(Similarity_TF)]), 1],
			   [np.min([np.min(Similarity_OG), np.min(Similarity_TF)]), 1], 'k-', dashes=[4, 2])

	ax[2].scatter(Similarity_SE.flatten(), Similarity_TF.flatten(), c=mc, alpha=0.6)
	ax[2].set_xlabel('Simple Elastix CC')
	ax[2].set_ylabel('STN CC')
	ax[2].plot([np.min([np.min(Similarity_SE), np.min(Similarity_TF)]), 1],
			   [np.min([np.min(Similarity_SE), np.min(Similarity_TF)]), 1], 'k-', dashes=[4, 2])

	plt.savefig("Elastix_Error.pdf", format='pdf')
	plt.show()

	print(elapsed_time_tf)
	print(elapsed_time_se)
	print(elapsed_time_se / elapsed_time_tf)
