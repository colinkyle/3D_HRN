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

params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_265x257_512000',
		  'save_file': 'regNETshallow',
		  'save_interval': 1000,
		  'batch_size': 10,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.9,  # RMS Prop decay
		  'rms_eps': 1e-8,  # RMS Prop epsilon
		  'width': 265,
		  'height': 257,
		  'numParam': 3,
		  'train': True}
netR = AtlasNet(params)

# retrain

if 0:
	files = fsys.file('D:/__Atlas__/data/35717/histology/segmented/*.png')[0:params['batch_size']]
	batch = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2), dtype=np.float32)
	for _ in range(5000):
		ideal = np.random.uniform(low=-20, high=20, size=(params['batch_size'], params['numParam']))
		ideal2 = copy.deepcopy(ideal)
		for b, f in enumerate(files):
			fixed = Img.imread(f)  # .p_intensity
			moving = Img.imread(f)  # .p_intensity
			batch[b, :, :, :] = np.stack((fixed, moving), axis=2)

		# ideal[-1,:] = [25,-25,5]
		moved, xytheta, theta0, cost = netR.transform_batch(batch, ideal)

		for i in range(params['batch_size']):
			batch[i, :, :, 1] = np.squeeze(moved[i, :, :])
			# calculate inverse transform
			ideal2[i] = calc_inverse_xform(ideal)
			# rot = affine2d_from_xytheta([ideal[i,0],ideal[i,1],-ideal[i,2]])[0:2]
			# pos = np.array([ideal[i,0],ideal[i,1],1])
			# dxy2 = pos[0:2] - np.sum(rot*pos,axis=1)
			# ideal2[i] = [dxy2[0],dxy2[1],-ideal[i,2]]

		# tformed, xytheta,theta2, cost = netR.transform_batch(batch, ideal2)
		# E = Edge(img0=Img.from_array(fixed), img1=Img.from_array(tformed[-1, :, :, :].squeeze()))
		# E.view(img2=moved[-1, :, :, :].squeeze())

		cnt, cost, out, cost_t, tfor = netR.train(batch, ideal2)
		print('count: {}, cost: {}, cost_t: {}'.format(cnt, cost, cost_t))
		if (params['save_file']):
			if cnt % params['save_interval'] == 0:
				netR.save_ckpt('model_saves/model-' + params['save_file'] + "_" + str(params['width']) + 'x' + str(
					params['height']) + '_' + str(cnt))
				print('Model saved')

if 1:
	files = fsys.file('D:/__Atlas__/data/35717/histology/segmented/*.png')  # [0:params['batch_size']]
	batch = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2), dtype=np.float32)
	# fig,axes = plt.subplots(3,3,figsize = (15,15))
	# for s in range(3):
	ErrorX = np.zeros((32, 10))
	ErrorY = np.zeros((32, 10))
	ErrorZ = np.zeros((32, 10))

	ErrorX_cv = np.zeros((32, 10))
	ErrorY_cv = np.zeros((32, 10))
	ErrorZ_cv = np.zeros((32, 10))

	XXX = np.zeros((32, 10))
	YYY = np.zeros((32, 10))
	ZZZ = np.zeros((32, 10))

	Similarity_OG = np.zeros((32, 10))
	Similarity_TF = np.zeros((32, 10))
	Similarity_SITK = np.zeros((32, 10))

	Time_cv2 = np.zeros((32, 10))

	cnt = 0
	elapsed_time_cv2 = 0
	elapsed_time_tf = 0
	for im in np.arange(0, 32):
		fixed = Img.imread(files[im])
		moving = Img.imread(files[im])
		batch = np.zeros((10, 265, 257, 2))
		for b in range(10):
			batch[b, :, :, :] = np.stack((fixed, moving), axis=2)
		batch2 = copy.deepcopy(batch)

		Xform = np.random.uniform(low=-20, high=20, size=(10, 3))
		Xform_inv = calc_inverse_xform(Xform)
		CV2_Error = copy.deepcopy(Xform)
		moved, xytheta0, theta0, cost = netR.transform_batch(batch, Xform)

		for i in range(params['batch_size']):
			# prep TF batch
			batch2[i, :, :, 1] = np.squeeze(moved[i, :, :])
			Similarity_OG[im, i] = ecc(batch2[i, :, :, 0], batch2[i, :, :, 1])
			# do cv2 registration
			# sz = moved[i, :, :].shape
			# # Define the motion model
			# warp_mode = cv2.MOTION_EUCLIDEAN
			# warp_matrix = np.eye(2, 3, dtype=np.float32)
			# number_of_iterations = 5000
			# termination_eps = 1e-5
			# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
			# start_time = time.time()
			# (cc, warp_matrix) = cv2.findTransformECC(fixed, moved[i, :, :], warp_matrix, warp_mode, criteria)
			# end_time = time.time()
			# Time_cv2[im,i] = end_time-start_time
			# elapsed_time_cv2 += end_time-start_time
			# warp_matrix = np.append(warp_matrix,[[0,0,1]],axis=0)
			# # calculate error
			# d_theta0 = math.atan2(warp_matrix[0, 1], warp_matrix[0, 0]) * 180 / math.pi
			# d_x0, d_y0, _ = np.sum(np.array([265. / 2, 257. / 2, 1]) * warp_matrix, axis=1) - np.array([265. / 2, 257. / 2, 1])
			#
			# CV2_Error[i] = [np.abs(d_x0-Xform_inv[i][0]),np.abs(d_y0-Xform_inv[i][1]),np.abs(d_theta0-Xform_inv[i][2])]
			# print(CV2_Error[i])
			# SITK registration
			elastixImageFilter = sitk.ElastixImageFilter()
			elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(batch2[i, :, :, 0]))
			elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moved[i, :, :].squeeze()))
			elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
			start_time = time.time()
			elastixImageFilter.Execute()
			end_time = time.time()
			Time_cv2[im, i] = end_time - start_time
			elapsed_time_cv2 += end_time - start_time

			SE_Out = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
			Similarity_SITK[im, i] = ecc(batch2[i, :, :, 0], SE_Out)

		start_time = time.time()
		tformed, xytheta1, cost_cc = netR.run_batch(batch2)
		end_time = time.time()
		elapsed_time_tf += end_time - start_time
		for i in range(tformed.shape[0]):
			Similarity_TF[im, i] = ecc(batch2[i, :, :, 0], tformed[i, :, :])

		# error = np.abs(Xform_inv-xytheta1)
		# error_diff = error-CV2_Error
		# ErrorX[cnt, :] = error_diff[:, 0]
		# ErrorY[cnt, :] = error_diff[:, 1]
		# ErrorZ[cnt, :] = error_diff[:, 2]
		#
		# ErrorX_cv[cnt, :] = CV2_Error[:, 0]
		# ErrorY_cv[cnt, :] = CV2_Error[:, 1]
		# ErrorZ_cv[cnt, :] = CV2_Error[:, 2]
		#
		# XXX[cnt,:] = np.abs(Xform_inv[:,0])
		# YYY[cnt, :] = np.abs(Xform_inv[:, 1])
		# ZZZ[cnt, :] = np.abs(Xform_inv[:, 2])
		# cnt+=1
		print(elapsed_time_cv2 - elapsed_time_tf)

	#     axes[0,s].boxplot(ErrorX,positions= np.arange(0,20))
	#     axes[0,s].set_title('X error (pixels)')
	#     axes[1,s].boxplot(ErrorY,positions= np.arange(0,20))
	#     axes[1,s].set_title('Y error (pixels)')
	#     axes[2,s].boxplot(ErrorZ,positions= np.arange(0,20))
	#     axes[2,s].set_title('Theta error (deg)')
	# plt.savefig("rigid_errors.pdf", format='pdf')
	# plt.show()

# print(ErrorX)

# cnt,cost, out, cost_t, tformed = netR.train(batch,ideal)
# moving.view()
# plt.show()
print(Time_cv2)

mc = [88. / 255, 184. / 255, 249. / 255]
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
ax[0].scatter(Similarity_OG.flatten(), Similarity_SITK.flatten(), c=mc, alpha=0.6)
ax[0].set_xlabel('Starting CC')
ax[0].set_ylabel('Simple Elastix CC')
ax[0].plot([np.min([np.min(Similarity_OG), np.min(Similarity_SITK)]), 1],
		   [np.min([np.min(Similarity_OG), np.min(Similarity_SITK)]), 1], 'k-', dashes=[4, 2])

ax[1].scatter(Similarity_OG.flatten(), Similarity_TF.flatten(), c=mc, alpha=0.6)
ax[1].set_xlabel('Starting CC')
ax[1].set_ylabel('STN CC')
ax[1].plot([np.min([np.min(Similarity_OG), np.min(Similarity_TF)]), 1],
		   [np.min([np.min(Similarity_OG), np.min(Similarity_TF)]), 1], 'k-', dashes=[4, 2])

ax[2].scatter(Similarity_SITK.flatten(), Similarity_TF.flatten(), c=mc, alpha=0.6)
ax[2].set_xlabel('Simple Elastix CC')
ax[2].set_ylabel('STN CC')
ax[2].plot([np.min([np.min(Similarity_SITK), np.min(Similarity_TF)]), 1],
		   [np.min([np.min(Similarity_SITK), np.min(Similarity_TF)]), 1], 'k-', dashes=[4, 2])

plt.savefig("Elastix_Rigid_Error.pdf", format='pdf')
plt.show()

print(elapsed_time_tf)
print(elapsed_time_se)
print(elapsed_time_se / elapsed_time_tf)

stop
mc = [88. / 255, 184. / 255, 249. / 255]
fig, ax = plt.subplots(3, 2, figsize=(10, 15))
ax[0, 1].scatter(Time_cv2.flatten(), ErrorX.flatten(), c=mc, alpha=0.2)
ax[0, 1].set_xlabel('time OpenCV')
ax[0, 1].set_ylabel('Error_STN - Error_OCV')
ax[0, 1].set_title(r'Error $\Delta$X')
ax[0, 1].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

ax[1, 1].scatter(Time_cv2.flatten(), ErrorY.flatten(), c=mc, alpha=0.2)
ax[1, 1].set_xlabel('time OpenCV')
ax[1, 1].set_ylabel('Error_STN - Error_OCV')
ax[1, 1].set_title(r'Error $\Delta$Y')
ax[1, 1].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

ax[2, 1].scatter(Time_cv2.flatten(), ErrorZ.flatten(), c=mc, alpha=0.2)
ax[2, 1].set_xlabel('time OpenCV')
ax[2, 1].set_ylabel('Error_STN - Error_OCV')
ax[2, 1].set_title(r'Error $\Delta \theta$')
ax[2, 1].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

ax[0, 0].scatter(Time_cv2.flatten(), ErrorX_cv.flatten(), c=mc, alpha=0.2)
ax[0, 0].set_xlabel('time OpenCV')
ax[0, 0].set_ylabel('Error_OCV')
ax[0, 0].set_title(r'Error $\Delta$X')
ax[0, 0].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

ax[1, 0].scatter(Time_cv2.flatten(), ErrorY_cv.flatten(), c=mc, alpha=0.2)
ax[1, 0].set_xlabel('time OpenCV')
ax[1, 0].set_ylabel('Error_OCV')
ax[1, 0].set_title(r'Error $\Delta$Y')
ax[1, 0].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

ax[2, 0].scatter(Time_cv2.flatten(), ErrorZ_cv.flatten(), c=mc, alpha=0.2)
ax[2, 0].set_xlabel('time OpenCV')
ax[2, 0].set_ylabel('Error_OCV')
ax[2, 0].set_title(r'Error $\Delta \theta$')
ax[2, 0].plot([0, np.max(Time_cv2)], [0, 0], 'k-', dashes=[4, 2])

plt.savefig("Reg_Error3.pdf", format='pdf')
plt.show()

print(elapsed_time_tf)
print(elapsed_time_cv2)
print(elapsed_time_cv2 / elapsed_time_tf)
