import copy
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import nibabel as nib
import numpy as np
import pickle
import pymc3 as pm
from pymc3.distributions.dist_math import alltrue_elemwise
# import pymp
import SimpleITK as sitk
from scipy import misc
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy import stats
from TF_models import RigidNet, TpsNet
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import tifffile
import fsys
import tensorflow as tf
import tqdm
import re
# debug
from theano.compile.nanguardmode import NanGuardMode


# TODO
# fix deepcopy on Img class
# CSVs with imgname, position, resolution
# deal with image background extraction
# test background extraction
# add cc to edge class


# this decorator allows for lazy computations
def lazy_property(fn):
	'''Decorator that makes a property lazy-evaluated.
	'''
	attr_name = '_lazy_' + fn.__name__

	@property
	def _lazy_property(self):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, fn(self))
		return getattr(self, attr_name)

	return _lazy_property


# main image class
class Img(np.ndarray):
	def __new__(cls, *args, **kwargs):
		meta = kwargs.pop('metadata', {})
		new = np.ndarray.__new__(cls, *args, **kwargs)
		new.meta = meta
		new.warped = np.zeros(new.shape,dtype=new.dtype)
		new.xytheta = np.zeros((1,3),dtype=np.float32)
		new.splines = np.zeros((1,128),dtype=np.float32)
		return new

	def __reduce__(self):
		# Get the parent's __reduce__ tuple
		pickled_state = super(Img, self).__reduce__()
		# Create our own tuple to pass to __setstate__
		new_state = pickled_state[2] + (self.meta,)
		# Return a tuple that replaces the parent's __setstate__ tuple with our own
		return (pickled_state[0], pickled_state[1], new_state)

	def __setstate__(self, state):
		self.meta = state[-1]  # Set the info attribute
		# Call the parent's __setstate__ with the other tuple elements.
		super(Img, self).__setstate__(state[0:-1])

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, copy.deepcopy(v, memo))
		return result

	@classmethod
	def imread(cls, fpath, metadata=None):
		[path, name, ext] = fsys.file_parts(fpath)
		if ext == '.tiff':
			# read META data
			with tifffile.TiffFile(fpath) as tif:
				data = tif.asarray()
				data[data == 255.0] = 0
				data = data / 255.0
				data = rgb2gray(data)
				# image data from file overrides passed metadata
				if hasattr(tif[0], 'image_description'):
					metadata = tif[0].image_description
					metadata = json.loads(metadata.decode('utf-8'))
				elif metadata:
					pass
				else:
					metadata = get_defaultMETA()
			return cls.from_array(data, metadata)
		elif ext == '.png':
			data = misc.imread(fpath)
			data[data == 255.0] = 0
			data = data / 255.0
			return cls.from_array(data, get_defaultMETA())

	@classmethod
	def from_array(cls, imgdat, metadata={}):
		if not metadata:
			metadata = get_defaultMETA()
		return cls(
			shape=imgdat.shape,
			buffer=imgdat.astype(np.float32),
			dtype=np.float32,
			metadata=metadata)

	def imwrite(self, fpath):
		metadata = json.dumps(self.meta)
		tifffile.imsave(fpath, self, description=metadata)

	def view(self):
		plt.imshow(np.ndarray(
			shape=self.shape,
			buffer=self.astype(np.float32),
			dtype=np.float32), cmap='gray', interpolation='bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()

	def compose_translation_3D(self, xyz):
		affine = np.array([1, 0, 0, xyz[0]], [0, 1, 0, xyz[1]], [0, 0, 1, xyz[2]], [0, 0, 0, 1])
		return np.dot(np.array(self.meta['MAT']), affine)

	def compose_rotation_3D(self, theta_xyz):
		affine_x = np.array([[1, 0, 0, 0], ...
		[0, math.cos(math.radians(theta_xyz[0])), -1. * math.sin(math.radians(theta_xyz[0])), 0], ...
							 [0, math.sin(math.radians(theta_xyz[0])), math.cos(math.radians(theta_xyz[0])), 0], ...
							 [0, 0, 0, 1]])

		affine_y = np.array([[math.cos(math.radians(theta_xyz[1])), 0, math.sin(math.radians(theta_xyz[1])), 0], ...
		[0, 1, 0, 0], ...
							 [-1. * math.sin(math.radians(theta_xyz[1])), 0, math.cos(math.radians(theta_xyz[1])), 0],
							 ...
							 [0, 0, 0, 1]])

		affine_z = np.array(
			[[math.cos(math.radians(theta_xyz[2])), -1. * math.sin(math.radians(theta_xyz[2])), 0, 0], ...
			[math.sin(math.radians(theta_xyz[2])), math.cos(math.radians(theta_xyz[2])), 0, 0], ...
			 [0, 0, 1, 0], ...
			 [0, 0, 0, 1]])

		affine = np.dot(np.dot(affine_x, affine_y), affine_z)
		return np.dot(np.array(self.meta['MAT']), affine)

	def compose_rigid_2D(self, xytheta):
		# affine_og = self.get_affine_2D()
		affine = np.array([[math.cos(math.radians(xytheta[2])), math.sin(math.radians(xytheta[2])), xytheta[0]],
						   [-1.0 * math.sin(math.radians(xytheta[2])), math.cos(math.radians(xytheta[2])), xytheta[1]],
						   [0, 0, 1]])

		return affine  # np.dot(affine_og, np.dot(affine_rot, affine_trans))

	def set_affine(self, MAT):
		self.meta['MAT'] = MAT

	def get_affine_2D(self):
		affine = self.meta['MAT']
		row1 = [affine[0][i] for i in [0, 1, 3]]
		row2 = [affine[1][i] for i in [0, 1, 3]]
		row3 = [0, 0, 1]
		return np.array([row1, row2, row3])

	@lazy_property
	def p_intensity(self):
		# Get all relatives
		binCount, bins = np.histogram(self, bins=256, range=[machineEpsilon(np.float32), 1.])
		binCount = binCount / max(binCount)
		p_intensity = np.zeros(shape=self.shape[:2], dtype=np.float32)
		l = bins[0]
		ibin = 0
		for h in bins[1:]:
			p_intensity[np.logical_and(self >= l, self < h)] = binCount[ibin]
			ibin += 1
			l = h
		p_intensity[self == h] = binCount[-1]
		# p_intensity = p_intensity/np.max(p_intensity)
		return p_intensity


# defines relationship between images in terms of
# probability distributions of relative positions
class Edge:
	def __init__(self, img0, img1):
		if (not isinstance(img0, Img)) or (not isinstance(img1, Img)):
			raise TypeError("Edge must reference two Img objects")
		self.node0 = img0
		self.node1 = img1

		self.dz = img1.meta['MAT0'][2][3] - img0.meta['MAT0'][2][3]

		self.param_x = [-self.node1.shape[0] / 2, 0, self.node1.shape[0] / 2]
		self.param_y = [0, 6]
		self.param_theta = [0, 6]

	def p_x(self, dx):
		px = self.param_x[2]*stats.laplace.pdf(dx, loc=self.param_x[0], scale=self.param_x[1])
		return px

	def p_y(self, dy):
		py = self.param_y[2]*np.power(stats.hypsecant.pdf(dy, loc=self.param_y[0], scale=self.param_y[1]), 0.1)
		return py

	def p_theta(self, dtheta):
		ptheta = self.param_theta[2] * np.power(stats.hypsecant.pdf(dtheta, loc=self.param_theta[0], scale=self.param_theta[1]), 0.01)
		return ptheta

	def sample_xytheta(self, nsamples=500):
		x = np.random.triangular(self.param_x[0], self.param_x[1], self.param_x[2], size=nsamples)

		y = np.random.normal(self.param_y[0], self.param_y[1], size=nsamples)

		theta = np.random.laplace(self.param_theta[0], self.param_theta[1], size=nsamples)

		return [x, y, theta]

	def view(self, img0=None, img1=None, img2=None):
		if img0 is None:
			img0 = self.node0
		else:
			img0 = img0 / max(img0.flatten())

		if img1 is None:
			img1 = self.node1
		else:
			img1 = img1 / max(img1.flatten())

		if img2 is None:
			img2 = np.zeros(self.node1.shape)
		else:
			img2 = img2 / max(img2.flatten())

		sz0 = img0.shape
		sz1 = img1.shape
		sz2 = img2.shape

		big_size = [max([sz0[0], sz1[0], sz2[0]]), max([sz0[1], sz1[1], sz2[1]]), 3]

		new_img = np.zeros(big_size)
		new_img[0:sz0[0], 0:sz0[1], 0] = img0
		new_img[0:sz1[0], 0:sz1[1], 1] = img1
		new_img[0:sz2[0], 0:sz2[1], 2] = img2

		plt.imshow(np.ndarray(
			shape=big_size,
			buffer=new_img.astype(np.float32),
			dtype=np.float32), cmap='gray', interpolation='bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()


# collection class
class Img3D:
	def __init__(self, fList=[], metaList=[]):
		"""Img3D is container of Img objects, fnames, and edge dictionary"""
		self.fList = fList
		self.images = []
		self.edges = {}

		if self.fList:
			for i, fname in enumerate(self.fList):
				self.images.append(Img.imread(fname))
				self.images[-1].meta['fname'] = fname
		if metaList:
			self.update_meta(metaList)

	def save(self, fname):
		save_obj(self, fname)

	def update_meta(self, metaList):
		if len(metaList) == len(self.images):
			for i, meta in enumerate(metaList):
				self.images[i].meta.update(meta)
		else:
			raise ValueError('metaList must be same length as self.images')

	def build_edge_net(self, n_neighbors):
		for i in range(len(self.images) - n_neighbors):
			for j in np.arange(i + 1, i + n_neighbors + 1):
				self.edges[(i, j)] = Edge(self.images[i], self.images[j])

	def estimate_edge_distributions(self, nsamples=1000):

		params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_257x265_507000',
				  'save_file': 'regNETshallow',
				  'save_interval': 1000,
				  'batch_size': 90,
				  'lr': .0001,  # Learning rate
				  'rms_decay': 0.9,  # RMS Prop decay
				  'rms_eps': 1e-8,  # RMS Prop epsilon
				  'width': 265,
				  'height': 257,
				  'numParam': 3,
				  'train': True}
		netR = AtlasNet(params)
		# Loop edges

		for edge_id, edge in self.edges.items():
			print(edge_id)
			# Search for best registration
			(cc, warp_cv2, xytheta_TF, moved) = self.rigid_reg(edge_id,netR)

			# tf.reset_default_graph()
			# del netR

			## alternate distribution method
			fixed, moving = self.edges[edge_id].node0, self.edges[edge_id].node1
			grid_size = 15
			batch_size = 75
			Xform = np.zeros((grid_size ** 3, 3))
			cnt = 0
			r = [30, 30, 10]
			num = 31
			for i in np.linspace(-r[0], r[0], grid_size):
				for j in np.linspace(-r[1], r[1], grid_size):
					for k in np.linspace(-r[2], r[2], grid_size):
						Xform[cnt] = [i+xytheta_TF[0,0], j+xytheta_TF[0,1], k+xytheta_TF[0,2]]
						cnt += 1

			batch = np.zeros((batch_size, 265, 257, 2))
			for i in range(batch_size):
				batch[i] = np.stack((fixed, moving), axis=2)

			CC = np.zeros((grid_size, grid_size, grid_size))
			y, x, z = np.meshgrid(range(grid_size), range(grid_size), range(grid_size))
			y = y.flatten()
			x = x.flatten()
			z = z.flatten()

			for b in range(int((grid_size**3)/batch_size)):
				#print(Xform[b * 80:b * 80 + 80])
				tformed, xytheta, theta2, cost_cc = netR.transform_batch(batch, Xform[b * batch_size:b * batch_size + batch_size])
				X, Y, Z = x[b * batch_size:b * batch_size + batch_size], y[b * batch_size:b * batch_size + batch_size], z[b * batch_size:b * batch_size + batch_size]
				for i in range(tformed.shape[0]):
					CC[X[i], Y[i], Z[i]] = ecc(fixed, tformed[i])

			dist_x = np.max(np.max(CC, axis=2, keepdims=True), axis=1, keepdims=True).flatten()
			dist_y = np.max(np.max(CC, axis=2, keepdims=True), axis=0, keepdims=True).flatten()
			dist_theta = np.max(np.max(CC, axis=1, keepdims=True), axis=0, keepdims=True).flatten()
			dist_x = np.concatenate((dist_x, [cc]), axis=0)
			dist_y = np.concatenate((dist_y, [cc]), axis=0)
			dist_theta = np.concatenate((dist_theta, [cc]), axis=0)

			x = np.linspace(-r[0], r[0], grid_size)
			y = np.linspace(-r[1], r[1], grid_size)
			z = np.linspace(-r[2], r[2], grid_size)
			x = np.concatenate([x, [0]], axis=0)
			y = np.concatenate([y, [0]], axis=0)
			z = np.concatenate([z, [0]], axis=0)
			x+=xytheta_TF[0,0]
			y+=xytheta_TF[0,1]
			z+=xytheta_TF[0,2]
			# reorder
			reord = np.argsort(x)
			x = x[reord]
			dist_x = dist_x[reord]

			reord = np.argsort(z)
			z = z[reord]
			dist_theta = dist_theta[reord]

			reord = np.argsort(y)
			y = y[reord]
			dist_y = dist_y[reord]

			edge.param_x = fit_laplace(x, dist_x)
			edge.param_y = fit_pr_hypsecant(y, dist_y, .1)
			edge.param_theta = fit_pr_hypsecant(z, dist_theta, .01)

			#plotting
			#dist_x_hat = stats.laplace.pdf(x, loc=u_x, scale=s_x)
			#dist_x_hat = np.max(dist_x) * dist_x_hat / np.max(dist_x_hat)
			#
			#dist_y_hat = np.power(stats.hypsecant.pdf(y, loc=u_y, scale=s_y), .1)
			#dist_y_hat = np.max(dist_y) * dist_y_hat / np.max(dist_y_hat)
			#
			#dist_z_hat = np.power(stats.hypsecant.pdf(z, loc=u_z, scale=s_z), .01)
			#dist_z_hat = np.max(dist_theta) * dist_z_hat / np.max(dist_z_hat)
			#
			#plt.plot(x,dist_x)
			#plt.plot(x,dist_x_hat)
			#plt.show()
			#
			#plt.plot(y, dist_y)
			#plt.plot(y, dist_y_hat)
			#plt.show()
			#
			#plt.plot(z, dist_theta)
			#plt.plot(z, dist_z_hat)
			#plt.show()



			# dy,dx,dtheta = warp[0][2],warp[1][2],np.arcsin(warp[0][1])*180/np.pi
			# print('max: ', dx,dy,dtheta,'cc: ',cc)
			#
			# # Generate to model full distribution around rigid warp parameters
			# X,Y,Theta = edge.sample_xytheta(nsamples=nsamples)
			# p_xytheta = np.zeros((nsamples,))
			# #with pymp.Parallel(4) as p:
			# for index in range(nsamples):
			#   transMAT = edge.node1.compose_rigid_2D([Y[index], X[index], Theta[index]])
			#   rows, cols = edge.node1.shape
			#   node1_transformed = cv2.warpAffine(edge.node1.p_intensity, transMAT[0:2, :], (cols, rows))
			#   p_xytheta[index] = ecc(node1_transformed, self.edges[edge_id].node0.p_intensity)
			#
			# # clean up in case nan
			# X = X[np.isfinite(p_xytheta)]
			# Y = Y[np.isfinite(p_xytheta)]
			# Theta = Theta[np.isfinite(p_xytheta)]
			# p_xytheta = p_xytheta[np.isfinite(p_xytheta)]
			#
			# # Model distribution
			# X1 = np.array(X, dtype=theano.config.floatX)
			# Y1 = np.array(Y, dtype=theano.config.floatX)
			# Theta1 = np.array(Theta, dtype=theano.config.floatX)
			#
			# Error = model_tgl([X1,Y1,Theta1],p_xytheta,[dx,dy,dtheta],cc,edge)
			# print('Error',Error)
			#
			# # Yhat = tpdf(X1,edge.param_x[0],edge.param_x[1],edge.param_x[2]).eval()*tpdf(Y1,edge.param_y[0],edge.param_y[1],edge.param_y[2]).eval() * lpdf(Theta1,edge.param_theta[0],edge.param_theta[1]).eval()
			# # fig = plt.figure()
			# # ax = fig.add_subplot(111, projection='3d')
			# # ax.scatter(X, Y, p_xytheta)
			# # ax.scatter(X, Y, Yhat,c='r')
			# # ax.set_xlabel('X Label')
			# # ax.set_ylabel('Y Label')
			# # ax.set_zlabel('P Label')
			# # plt.show()
			self.save('TestImg3D.obj')

	def rigid_reg(self, edgeID,netR):
		(cc, warp_cv2, xytheta_TF, moved) = rigid_reg(self.edges[edgeID].node0, self.edges[edgeID].node1,netR)
		return (cc, warp_cv2, xytheta_TF, moved)

	def score_param_consistency(self, xytheta):
		half_width = self.images[0].shape[1] / 2.
		half_height = self.images[0].shape[0] / 2.
		total_err = 0
		for pair in self.edges.keys():
			# convert center rotated to origin rotated
			warp1 = affine2d_from_xytheta(xytheta[pair[0]])  # img1 warp
			warp2 = affine2d_from_xytheta(xytheta[pair[1]])  # img2 warp

			# calculate img1 -> img0
			warp3 = np.dot(warp1, np.linalg.inv(warp2))
			_, _, mri_tz, mri_x, mri_y, _ = affine_get_params(affine_2d_to_3d(warp3))

			# get probabilities from edge distributions
			p_x = self.edges[pair].p_x(mri_x)
			p_y = self.edges[pair].p_y(mri_y)
			p_theta = self.edges[pair].p_theta(mri_tz)
			# cost function
			p_edge = p_x * p_y * p_theta * (1 - .05 ** np.diff(pair))

			if not np.isnan(p_edge):
				total_err += np.log(p_edge)
		return np.exp(total_err / len(self.edges.keys()))

	def set_warped_from_tformed(self,tformed,xytheta=None,splines=None):
		if len(self.images) != tformed.shape[0]:
			raise ValueError('tformed.shape[0] must be same length as self.images')
		for i in range(len(self.images)):
			self.images[i].warped = tformed[i].squeeze()
		if xytheta is not None:
			for i in range(len(self.images)):
				self.images[i].xytheta = xytheta[i]
		if splines is not None:
			for i in range(len(self.images)):
				self.images[i].splines = splines[i]

	def get_mri_from_warped(self):
		locs = [int(re.search('[0-9]*[N]', self.fList[i]).group(0)[0:-1]) for i in range(len(self.fList)) if
				re.search('[0-9]*[N]', self.fList[i])]
		z_middle = (max(locs) + min(locs)) / 2.

		z_all = np.arange(min(locs), max(locs), 1)
		z_arg = np.empty(z_all.shape)
		z_arg[:] = None

		for i, f in enumerate(self.fList):
			z_pos = int(re.search('[0-9]*[N]', f).group(0)[0:-1])
			arg_loc = np.argwhere(z_all == z_pos)
			z_arg[arg_loc] = i

		hist_data = np.zeros((265, 257, len(z_arg)))
		for z in range(len(z_arg)):
			if ~np.isnan(z_arg[z]):
				hist_data[:, :, z] = self.images[np.int(z_arg[z])].warped

		affine = np.array(copy.deepcopy(self.images[0].meta['MAT']))
		affine[2, 3] = -affine[2, 2] * ((max(locs) - min(locs)) / 2.)
		return nib.Nifti1Image(hist_data, affine)


class MRI:

	def __init__(self, fname):

		self.affine = nib.load(fname).affine
		self.data = nib.load(fname).get_data()
		self.data = self.data / max(self.data.flatten())

	@lazy_property
	def p_intensity(self):
		# Get all relatives
		binCount, bins = np.histogram(self.data, bins=256, range=[machineEpsilon(np.float32), 1.])
		binCount = binCount / max(binCount)
		p_intensity = np.zeros(shape=self.data.shape, dtype=np.float32)
		l = bins[0]
		ibin = 0
		for h in bins[1:]:
			p_intensity[np.logical_and(self.data >= l, self.data < h)] = binCount[ibin]
			ibin += 1
			l = h
		p_intensity[self.data == h] = binCount[-1]
		# p_intensity = p_intensity/np.max(p_intensity)
		return p_intensity

	def reg_to_slice(self, IMG, dz=0, theta_x=0, theta_y=0):
		warpCoef = []
		warpList = []
		# per section
		# with pymp.Parallel(4) as p:
		for i, img in enumerate(IMG.images):
			# x,y,z MRI
			x_mri = np.array([self.affine[0][0] * x + self.affine[0][3] for x in range(self.data.shape[0])])
			y_mri = np.array([self.affine[1][1] * y + self.affine[1][3] for y in range(self.data.shape[1])])
			z_mri = np.array([self.affine[2][2] * z + self.affine[2][3] for z in range(self.data.shape[2])])

			# scale and position of histology
			x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
			y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
			z = img.meta['MAT'][2][3] + dz
			XX, YY = np.atleast_2d(x, y)
			YY = YY.T

			# calculate rotation matrices
			affine_x = np.array([[1, 0, 0, 0],
								 [0, math.cos(math.radians(theta_x)), -1. * math.sin(math.radians(theta_x)), 0],
								 [0, math.sin(math.radians(theta_x)), math.cos(math.radians(theta_x)), 0],
								 [0, 0, 0, 1]])

			affine_y = np.array([[math.cos(math.radians(theta_y)), 0, math.sin(math.radians(theta_y)), 0],
								 [0, 1, 0, 0],
								 [-1. * math.sin(math.radians(theta_y)), 0, math.cos(math.radians(theta_y)), 0],
								 [0, 0, 0, 1]])
			MAT = np.dot(affine_x, affine_y)

			# transform sample points
			XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
			YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
			ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]

			# interpolate results
			mri = interpn((x_mri, y_mri, z_mri), self.p_intensity, np.array([XXX, YYY, ZZZ]).T, bounds_error=False,
						  fill_value=0)
			(cc, warp) = rigid_reg(mri.astype(np.float32), img.p_intensity)
			warpList.append(warp)
			warpCoef.append(cc)
			dy, dx, dtheta = warp[0][2], warp[1][2], np.arcsin(warp[0][1]) * 180 / np.pi
			print('max: ', dx, dy, dtheta, 'cc: ', cc)

		return (warpList, warpCoef)

	def retrain_TF(self, net, big_batch, ntrain=500, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, out, cost_t, _ = net.train(batch, np.zeros((nbatch, 3)))

			# print('count: {}, cost_cc: {}, sanity: {}'.format(cnt, cost_t, np.mean(np.abs(out), axis=0)))
		return net

	def retrain_TF_E(self, net, big_batch, ntrain=500, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, cost_cc, E_det_j, E_div, E_curl = net.train(batch)

			# print('count: {}, cost_cc: {}, sanity: {}'.format(cnt, cost_t, np.mean(np.abs(out), axis=0)))
		return net

	# helper functions

	def retrain_TF_Both(self, netR, netE, big_batch, ntrain=500, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, out, cost_t, tformed = netR.train(batch, np.zeros((nbatch, 3)))
			for i in range(nbatch):
				batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])
			cnt, cost, cost_cc, E_det_j, E_div, E_curl = netE.train(batch)

			# print('count: {}, cost_cc: {}, sanity: {}'.format(cnt, cost_t, np.mean(np.abs(out), axis=0)))
		return netR, netE

	# helper functions

	# def param_search(self, IMG, zlim, theta_xlim, theta_ylim, xy_res, z_res):
	#   # variables
	#   nspacing = 5
	#   nbatch = nspacing**5
	#   n_each_param = 1
	#   ntrain = [5000, 500, 200, 200]
	#   ## load net
	#   print('loading models...')
	#   # go to training dir
	#   fsys.cd('D:/Dropbox/__Atlas__')
	#   params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_257x265_507000',
	#             'save_file': 'regNETshallow',
	#             'save_interval': 1000,
	#             'batch_size': 32,
	#             'lr': .0001,  # Learning rate
	#             'rms_decay': 0.9,  # RMS Prop decay
	#             'rms_eps': 1e-8,  # RMS Prop epsilon
	#             'width': 265,
	#             'height': 257,
	#             'numParam': 3,
	#             'train': True}
	#   netR = AtlasNet(params)
	#   params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic2_257x265_1000'
	#   params['save_file'] = 'Elastic2'
	#   netE = ElasticNet(params)
	#
	#   param_dist = []
	#   max_score = 0
	#   max_param = [0, 0, 0]
	#   #loop parameter resolution
	#   for res in range(4):
	#     ## gen training batch
	#     print('\ngenerating training batch...')
	#
	#     z_vals = []#np.random.randint(zlim[0],zlim[1],nbatch)
	#     theta_x_vals = []
	#     theta_y_vals = []
	#     dxy_vals = []
	#     dz_vals = []
	#     zs = np.linspace(zlim[0],zlim[1], nspacing)
	#     txs = np.linspace(theta_xlim[0], theta_xlim[1], nspacing)
	#     tys = np.linspace(theta_ylim[0], theta_ylim[1], nspacing)
	#     dxys = np.linspace(xy_res[0], xy_res[1], nspacing)
	#     dzs = np.linspace(z_res[0], z_res[1], nspacing)
	#
	#     for z in zs:
	#       for tx in txs:
	#         for ty in tys:
	#           for dxy in dxys:
	#             for dz in dzs:
	#               z_vals.append(z)
	#               theta_x_vals.append(tx)
	#               theta_y_vals.append(ty)
	#               dxy_vals.append(dxy)
	#               dz_vals.append(dz)
	#
	#
	#     big_batch = np.zeros(shape=(n_each_param*nbatch,params['width'],params['height'],2),dtype=np.float32)
	#
	#     pos = (0,n_each_param)
	#     for z,tx,ty,dxy,dz in tqdm.tqdm(zip(z_vals,theta_x_vals,theta_y_vals,dxy_vals,dz_vals)):
	#       big_batch[pos[0]:pos[1],:,:,:] = self.gen_batch(IMG,z,tx,ty,dxy,dz,subsample=True,n_subsample=n_each_param)#[np.random.choice(range(len(IMG.images)),n_each_param),:,:,:]
	#       pos = (pos[0]+n_each_param,pos[1]+n_each_param)
	#
	#     # retrain networks
	#     if ntrain[res]>0:
	#       netR,netE = self.retrain_TF_Both(netR,netE,big_batch,ntrain=ntrain[res],nbatch=32)
	#
	#     ## compute fits
	#     print('\ncomputing parameter scores...')
	#     score = np.zeros(shape=(nbatch,))
	#     pos = 0
	#     for z,tx,ty,dxy,dz in tqdm.tqdm(zip(z_vals,theta_x_vals,theta_y_vals,dxy_vals,dz_vals)):
	#       batch = self.gen_batch(IMG,z,tx,ty,dxy,dz)
	#       #run rigid
	#       tformed, xytheta, _ = netR.run_batch(batch)
	#       for i in range(tformed.shape[0]):
	#         batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])
	#
	#       #run elastic
	#       tformed, theta, cost_cc, cost = netE.run_batch(batch)
	#       # compute global cost function
	#       p_consistency = IMG.score_param_consistency(xytheta)
	#       score[pos] = .4*np.mean(cost_cc) + .6*p_consistency - cost
	#       pos+=1
	#
	#     ## update parameter ranges
	#     param_dist.append(zip(z_vals,theta_x_vals,theta_y_vals,dxy_vals,dz_vals,score))
	#     plt.figure()
	#     n,bins,_ = plt.hist(score)
	#     plt.show()
	#     max_id = np.argmax(score)
	#     print('\nmax score:',np.max(score), 'pos:',z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id], dxy_vals[max_id], dz_vals[max_id])
	#
	#     if np.max(score)> max_score:
	#       max_score = np.max(score)
	#       max_param = [z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id], dxy_vals[max_id], dz_vals[max_id]]
	#       #update z
	#       z_span = np.asscalar(np.diff(zlim))/4.
	#       zlim = [z_vals[max_id] - z_span, z_vals[max_id] + z_span]
	#       # update theta x
	#       tx_span = np.asscalar(np.diff(theta_xlim)) / 4.
	#       theta_xlim = [theta_x_vals[max_id] - tx_span, theta_x_vals[max_id] + tx_span]
	#       # update theta y
	#       ty_span = np.asscalar(np.diff(theta_ylim)) / 4.
	#       theta_ylim = [theta_y_vals[max_id] - ty_span, theta_y_vals[max_id] + ty_span]
	#
	#       # update dxy
	#       dxy_span = np.asscalar(np.diff(xy_res)) / 4.
	#       xy_res = [dxy_vals[max_id] - dxy_span, dxy_vals[max_id] + dxy_span]
	#
	#       # update dz
	#       dz_span = np.asscalar(np.diff(z_res)) / 4.
	#       z_res = [dz_vals[max_id] - dz_span, dz_vals[max_id] + dz_span]
	#
	#   ## close net
	#   tf.reset_default_graph()
	#   del netR, netE
	#   return max_score,max_param,param_dist

	def param_search(self, IMG, zlim, theta_xlim, theta_ylim, xy_res):
		# variables
		nspacing = 8
		nbatch = nspacing ** 4
		n_each_param = 1
		ntrain = [0]#, 0, 0, 0]#[5000, 500, 200, 200]
		## load net
		print('loading models...')
		# go to training dir
		fsys.cd('D:/__Atlas__')
		params = {'load_file': 'D:/__Atlas__/model_saves/model-Rigid30890',
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
		netR = RigidNet(params)
		params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic30890'
		params['save_file'] = 'Elastic2'
		netE = TpsNet(params)

		param_dist = []
		max_score = 0
		max_param = [0, 0, 0]
		# loop parameter resolution
		for res in range(len(ntrain)):
			## gen training batch
			print('\ngenerating training batch...')

			z_vals = []  # np.random.randint(zlim[0],zlim[1],nbatch)
			theta_x_vals = []
			theta_y_vals = []
			dxy_vals = []

			zs = np.linspace(zlim[0], zlim[1], nspacing)
			txs = np.linspace(theta_xlim[0], theta_xlim[1], nspacing)
			tys = np.linspace(theta_ylim[0], theta_ylim[1], nspacing)
			dxys = np.linspace(xy_res[0], xy_res[1], nspacing)

			for z in zs:
				for tx in txs:
					for ty in tys:
						for dxy in dxys:
							z_vals.append(z)
							theta_x_vals.append(tx)
							theta_y_vals.append(ty)
							dxy_vals.append(dxy)
			if ntrain[res] > 0:
				big_batch = np.zeros(shape=(n_each_param * nbatch, params['width'], params['height'], 2), dtype=np.float32)

				pos = (0, n_each_param)
				for z, tx, ty, dxy in tqdm.tqdm(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals)):
					big_batch[pos[0]:pos[1], :, :, :] = self.gen_batch(IMG, z, tx, ty, dxy, dxy, subsample=True,
																	   n_subsample=n_each_param)  # [np.random.choice(range(len(IMG.images)),n_each_param),:,:,:]
					pos = (pos[0] + n_each_param, pos[1] + n_each_param)

			#view batch?
			# plt.imshow(np.concatenate([big_batch[0], np.zeros([265, 257, 1])], axis=2))
			# plt.show()
			# retrain networks

				netR, netE = self.retrain_TF_Both(netR, netE, big_batch, ntrain=ntrain[res], nbatch=32)

			## compute fits
			print('\ncomputing parameter scores...')
			score = np.zeros(shape=(nbatch,))
			each_score = np.zeros(shape=(nbatch, 3))
			pos = 0
			for z, tx, ty, dxy in tqdm.tqdm(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals)):
				batch = self.gen_batch(IMG, z, tx, ty, dxy, dxy)
				# run rigid
				tformed, xytheta, _ = netR.run(batch)
				for i in range(tformed.shape[0]):
					batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])

				# run elastic
				tformed, theta, cost_cc, cost, cost2 = netE.run(batch)
				# compute global cost function
				p_consistency = IMG.score_param_consistency(xytheta)
				score[pos] = .4 * np.mean(cost_cc) + .6 * p_consistency - cost2
				if np.isnan(score[pos]):
					score[pos] = 0

				each_score[pos, 0] = cost_cc
				each_score[pos, 1] = p_consistency
				each_score[pos, 2] = cost2
				pos += 1

			## update parameter ranges
			param_dist.append(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals, score, each_score.T.tolist()[0],
								  each_score.T.tolist()[1], each_score.T.tolist()[2]))
			plt.figure()
			n, bins, _ = plt.hist(score)
			plt.show()
			max_id = np.argmax(score)
			print('\nmax score:', np.max(score), 'pos:', z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id],
				  dxy_vals[max_id])

			if np.max(score) > max_score:
				max_score = np.max(score)
				max_param = [z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id], dxy_vals[max_id]]
				# update z
				z_span = np.asscalar(np.diff(zlim)) / 4.
				zlim = [z_vals[max_id] - z_span, z_vals[max_id] + z_span]
				# update theta x
				tx_span = np.asscalar(np.diff(theta_xlim)) / 4.
				theta_xlim = [theta_x_vals[max_id] - tx_span, theta_x_vals[max_id] + tx_span]
				# update theta y
				ty_span = np.asscalar(np.diff(theta_ylim)) / 4.
				theta_ylim = [theta_y_vals[max_id] - ty_span, theta_y_vals[max_id] + ty_span]

				# update dxy
				dxy_span = np.asscalar(np.diff(xy_res)) / 4.
				xy_res = [dxy_vals[max_id] - dxy_span, dxy_vals[max_id] + dxy_span]

		## close net
		tf.reset_default_graph()
		del netR, netE
		return max_score, max_param, param_dist

	def gen_batch(self, IMG, z_loc, theta_x, theta_y, dxy, dz, subsample=False, n_subsample=0):
		# set xyz resolution
		mri_affine = copy.deepcopy(self.affine)
		mri_affine[:2, :] = dxy * mri_affine[:2, :]
		mri_affine[2, :] = dz * mri_affine[2, :]
		# x,y,z MRI
		x_mri = np.array([mri_affine[0][0] * x + mri_affine[0][3] for x in range(self.data.shape[0])])
		y_mri = np.array([mri_affine[1][1] * y + mri_affine[1][3] for y in range(self.data.shape[1])])
		z_mri = np.array([mri_affine[2][2] * z + mri_affine[2][3] for z in range(self.data.shape[2])])

		affine_x = np.array([[1, 0, 0, 0],
							 [0, math.cos(math.radians(theta_x)), -1. * math.sin(math.radians(theta_x)), 0],
							 [0, math.sin(math.radians(theta_x)), math.cos(math.radians(theta_x)), 0],
							 [0, 0, 0, 1]])

		affine_y = np.array([[math.cos(math.radians(theta_y)), 0, math.sin(math.radians(theta_y)), 0],
							 [0, 1, 0, 0],
							 [-1. * math.sin(math.radians(theta_y)), 0, math.cos(math.radians(theta_y)), 0],
							 [0, 0, 0, 1]])
		MAT = np.dot(affine_x, affine_y)

		if subsample:
			batch = np.zeros(shape=(n_subsample, IMG.images[0].shape[0], IMG.images[0].shape[1], 2))
			samples = np.random.randint(0, len(IMG.images), n_subsample)
			for i, samp in enumerate(samples):
				# Match img to mri
				img = IMG.images[samp]
				x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
				y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
				z = img.meta['MAT'][2][3] + z_loc
				XX, YY = np.atleast_2d(x, y)
				YY = YY.T

				XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
				YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
				ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
				# interp_func = RegularGridInterpolator((x_mri, y_mri, z_mri),self.data,bounds_error=False,
				#             fill_value=0)
				# mri = Img.from_array(
				#     interp_func(np.array([XXX, YYY, ZZZ]).T))
				mri = Img.from_array(
					interpn((x_mri, y_mri, z_mri), self.data, np.array([XXX, YYY, ZZZ]).T, bounds_error=False,
							fill_value=0))
				# match intensities with sitk
				matcher = sitk.HistogramMatchingImageFilter()
				matcher.SetNumberOfHistogramLevels(512)
				matcher.SetNumberOfMatchPoints(30)
				img = matcher.Execute(sitk.GetImageFromArray(img, sitk.sitkFloat32),
									  sitk.GetImageFromArray(mri, sitk.sitkFloat32))
				img = sitk.GetArrayFromImage(img)
				# batch[i, :, :, :] = cv2.merge((np.rot90(mri.p_intensity), np.rot90(img.p_intensity)))
				batch[i, :, :, :] = cv2.merge((mri, img))
			return batch

		# per section
		batch = np.zeros(shape=(len(IMG.images), IMG.images[0].shape[0], IMG.images[0].shape[1], 2))
		for i, img in enumerate(IMG.images):
			x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
			y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
			z = img.meta['MAT'][2][3] + z_loc
			XX, YY = np.atleast_2d(x, y)
			YY = YY.T

			XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
			YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
			ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
			mri = Img.from_array(
				interpn((x_mri, y_mri, z_mri), self.data, np.array([XXX, YYY, ZZZ]).T,
						bounds_error=False, fill_value=0))
			matcher = sitk.HistogramMatchingImageFilter()
			matcher.SetNumberOfHistogramLevels(512)
			matcher.SetNumberOfMatchPoints(30)
			img = matcher.Execute(sitk.GetImageFromArray(img, sitk.sitkFloat32),
								  sitk.GetImageFromArray(mri, sitk.sitkFloat32))
			img = sitk.GetArrayFromImage(img)
			batch[i, :, :, :] = cv2.merge((mri, img))
		return batch

	def get_resampled_mri(self, IMG, z_loc, theta_x, theta_y, dxy, dz):
		mri_affine = copy.deepcopy(self.affine)
		mri_affine[:2, :] = dxy * mri_affine[:2, :]
		mri_affine[2, :] = dz * mri_affine[2, :]

		# x,y,z MRI
		x_mri = np.array([mri_affine[0][0] * x + mri_affine[0][3] for x in range(self.data.shape[0])])
		y_mri = np.array([mri_affine[1][1] * y + mri_affine[1][3] for y in range(self.data.shape[1])])
		z_mri = np.array([mri_affine[2][2] * z + mri_affine[2][3] for z in range(self.data.shape[2])])

		affine_x = np.array([[1, 0, 0, 0],
							 [0, math.cos(math.radians(theta_x)), -1. * math.sin(math.radians(theta_x)), 0],
							 [0, math.sin(math.radians(theta_x)), math.cos(math.radians(theta_x)), 0],
							 [0, 0, 0, 1]])

		affine_y = np.array([[math.cos(math.radians(theta_y)), 0, math.sin(math.radians(theta_y)), 0],
							 [0, 1, 0, 0],
							 [-1. * math.sin(math.radians(theta_y)), 0, math.cos(math.radians(theta_y)), 0],
							 [0, 0, 0, 1]])

		MAT = np.dot(affine_x, affine_y)
		# failed attempt
		# trans_z = np.eye(4)
		# trans_z[2,3] = z

		# size = np.eye(4)
		# size[0:3,0:3] = dxy*size[0:3,0:3]

		z_bounds_mri = (np.sum(mri_affine[2, :] * np.array([0, 0, 0, 1])),
						np.sum(mri_affine[2, :] * np.array([0, 0, self.data.shape[2], 1])))
		zs_mri = np.linspace(z_bounds_mri[0], z_bounds_mri[1], self.data.shape[2])

		new_mri = np.zeros((265, 257, self.data.shape[2]))

		for i, z2 in enumerate(zs_mri):
			x = np.array([IMG.images[0].meta['MAT'][0][0] * x + IMG.images[0].meta['MAT'][0][3] for x in
						  range(IMG.images[0].shape[0])])
			y = np.array([IMG.images[0].meta['MAT'][1][1] * y + IMG.images[0].meta['MAT'][1][3] for y in
						  range(IMG.images[0].shape[1])])
			z = z2 + z_loc
			XX, YY = np.atleast_2d(x, y)
			YY = YY.T

			XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
			YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
			ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
			new_mri[:, :, i] = interpn((x_mri, y_mri, z_mri), self.data, np.array([XXX, YYY, ZZZ]).T,
									   bounds_error=False, fill_value=0)

		affine = np.array(IMG.images[0].meta['MAT'])
		affine[2, 2] = np.diff(zs_mri)[0]
		affine[2, 3] = zs_mri[0]
		# MAT = np.dot(np.dot(np.dot(affine_x, affine_y),size),trans_z)
		# new_affine = np.dot(MAT,mri.affine)
		# new_affine
		return nib.Nifti1Image(new_mri, affine)



# for p_intensity to ignore zero/background
def machineEpsilon(func=float):
	machine_epsilon = func(1)
	while func(1) + func(machine_epsilon) != func(1):
		machine_epsilon_last = machine_epsilon
		machine_epsilon = func(machine_epsilon) / func(2)
	return machine_epsilon_last


# only handle grayscale images
def rgb2gray(rgb):
	if len(rgb.shape) > 2:
		r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
		gray = 0.2989 * r + 0.5870 * b + 0.1140 * g
		return gray
	else:
		return rgb


def get_defaultMETA():
	affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
	meta = {'MAT0': affine, 'MAT': copy.deepcopy(affine), 'fname': None, 'parent': None, 'Children': {}}
	return meta


# theano distributions
def tpdf(value, lower, c, upper):
	# return ifelse(T.gt(T.concatenate([T.gt(lower,c), T.lt(upper,c), T.gt(lower,upper)]).sum(),0),T.zeros_like(value),T.switch(alltrue_elemwise([lower <= value, value < c]),
	#                 ((value - lower) / ((c - lower))),
	#                 T.switch(T.eq(value, c), (1),
	#                 T.switch(alltrue_elemwise([c < value, value <= upper]),
	#                 ((upper - value) / ((upper - c))), 0))))

	return T.switch(alltrue_elemwise([lower <= value, value < c]),
					((value - lower) / ((c - lower) + np.array(1e-32, dtype=theano.config.floatX))),
					T.switch(T.eq(value, c), (1),
							 T.switch(alltrue_elemwise([c < value, value <= upper]),
									  ((upper - value) / ((upper - c) + np.array(1e-32, dtype=theano.config.floatX))),
									  0)))


def gpdf(sample, location, scale):
	divisor = 2 * scale ** 2
	exp_arg = -((sample - location) ** 2) / divisor
	return T.exp(exp_arg)


def lpdf(sample, location, scale):
	divisor = 2 * scale  # + epsilon,
	exp_arg = -T.abs_(sample - location) / divisor
	return T.exp(exp_arg)


# similarity function
def ecc(img0, img1):
	return np.dot(img0.flatten(), img1.flatten()) / np.linalg.norm(img0.flatten()) / np.linalg.norm(img1.flatten())


# pymc models
def model_ttl(locations, samples, centers, cc, edge):
	basic_model = pm.Model()

	with basic_model:
		# Priors for unknown model parameters
		l1 = pm.Normal('l1', mu=edge.param_x[0], sd=edge.node0.shape[0] / 85)
		m1 = centers[0]
		u1 = pm.Normal('u1', mu=edge.param_x[2], sd=edge.node0.shape[0] / 85)

		l2 = pm.Normal('l2', mu=edge.param_y[0], sd=edge.node0.shape[1] / 85)
		m2 = centers[1]
		u2 = pm.Normal('u2', mu=edge.param_y[2], sd=edge.node0.shape[1] / 85)

		m3 = centers[2]
		s3 = pm.HalfNormal('s3', sd=20)

		p_x = tpdf(locations[0], l1, m1, u1)
		p_y = tpdf(locations[1], l2, m2, u2)
		p_theta = lpdf(locations[2], m3, s3)

		sigma = pm.HalfNormal('sigma', sd=1)

		# Expected value of outcome
		mu = cc * p_x * p_y * p_theta

		# Likelihood (sampling distribution) of observations
		Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=samples)
		trace = pm.sample(2000, njobs=1, step=pm.Metropolis())

	pm.summary(trace)
	# values
	L1 = np.mean(trace['l1'])
	M1 = centers[0]
	U1 = np.mean(trace['u1'])

	L2 = np.mean(trace['l2'])
	M2 = centers[1]
	U2 = np.mean(trace['u2'])

	M3 = centers[2]
	S3 = np.mean(trace['s3'])

	p_x = tpdf(locations[0], L1, M1, U1).eval()
	p_y = tpdf(locations[1], L2, M2, U2).eval()
	p_theta = lpdf(locations[2], M3, S3).eval()
	mu = cc * p_x * p_y * p_theta
	Err = np.sum((samples - mu) ** 2)
	# print(Err)

	edge.param_x = [L1, M1, U1]
	edge.param_y = [L2, M2, U2]
	edge.param_theta = [M3, S3]
	return Err


def model_ggl(locations, samples, centers, cc):
	basic_model = pm.Model()
	with basic_model:
		# Priors for unknown model parameters
		s1 = pm.HalfNormal('s1', sd=20)
		m1 = centers[0]

		s2 = pm.Normal('s2', sd=20)
		m2 = centers[1]

		m3 = centers[2]
		s3 = pm.HalfNormal('s3', sd=20)

		p_x = gpdf(locations[0], m1, s1)
		p_y = gpdf(locations[1], m2, s2)
		p_theta = lpdf(locations[2], m3, s3)

		sigma = pm.HalfNormal('sigma', sd=1)

		# Expected value of outcome
		mu = cc * p_x * p_y * p_theta

		# Likelihood (sampling distribution) of observations
		Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=samples)
		trace = pm.sample(5000, njobs=4)

	pm.summary(trace)
	# values
	S1 = np.mean(trace['s1'])
	M1 = centers[0]

	S2 = np.mean(trace['s2'])
	M2 = centers[1]

	M3 = centers[2]
	S3 = np.mean(trace['s3'])

	p_x = gpdf(locations[0], M1, S1).eval()
	p_y = gpdf(locations[1], M2, S2).eval()
	p_theta = lpdf(locations[2], M3, S3).eval()
	mu = cc * p_x * p_y * p_theta
	Err = np.sum((samples - mu) ** 2)
	print(Err)

	# import matplotlib.pyplot as plt
	# from mpl_toolkits.mplot3d import Axes3D
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	#
	# ax.scatter(locations[0], locations[1], zs=samples)
	# ax.scatter(locations[0], locations[1], zs=mu, c='r')
	# ax.azim = 0
	# ax.elev = 0
	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_zlabel('P')
	# plt.show()
	#
	# exit()


def model_tgl(locations, samples, centers, cc, edge):
	basic_model = pm.Model()

	with basic_model:
		# Priors for unknown model parameters
		l1 = pm.Normal('l1', mu=edge.param_x[0], sd=edge.node0.shape[0] / 85)
		m1 = centers[0]
		u1 = pm.Normal('u1', mu=edge.param_x[2], sd=edge.node0.shape[0] / 85)

		s2 = pm.Normal('s2', sd=20)
		m2 = centers[1]

		m3 = centers[2]
		s3 = pm.HalfNormal('s3', sd=20)

		p_x = tpdf(locations[0], l1, m1, u1)
		p_y = gpdf(locations[1], m2, s2)
		p_theta = lpdf(locations[2], m3, s3)

		sigma = pm.HalfNormal('sigma', sd=1)

		# Expected value of outcome
		mu = cc * p_x * p_y * p_theta

		# Likelihood (sampling distribution) of observations
		Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=samples)
		trace = pm.sample(2000, njobs=1, step=pm.Metropolis())

	pm.summary(trace)
	# values
	L1 = np.mean(trace['l1'])
	M1 = centers[0]
	U1 = np.mean(trace['u1'])

	M2 = centers[1]
	S2 = np.mean(trace['s2'])

	M3 = centers[2]
	S3 = np.mean(trace['s3'])

	p_x = tpdf(locations[0], L1, M1, U1).eval()
	p_y = gpdf(locations[1], M2, S2).eval()
	p_theta = lpdf(locations[2], M3, S3).eval()
	mu = cc * p_x * p_y * p_theta
	Err = np.sum((samples - mu) ** 2)
	# print(Err)

	edge.param_x = [L1, M1, U1]
	edge.param_y = [M2, S2]
	edge.param_theta = [M3, S3]
	return Err


# object save functions
def save_obj(var, fname):
	file_handler = open(fname, 'wb')
	pickle.dump(var, file_handler)


def load_obj(fname):
	filehandler = open(fname, 'rb')
	return pickle.load(filehandler)


# CV2 rigid registration function
def rigid_reg(fixed, moving,netR):
	warp_mode = cv2.MOTION_EUCLIDEAN
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	# Specify the number of iterations.
	number_of_iterations = 5000

	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10

	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC(fixed, moving, warp_matrix, warp_mode, criteria)
	## v1 note: x,y,transform directions are reversed in cv2 vs TF
	warp_cv2 = np.eye(3)
	warp_cv2[0:2] = warp_matrix
	warp_cv2 = np.linalg.inv(warp_cv2)[0:2]
	rows, cols = moving.shape
	# node1_transformed = cv2.warpAffine(moving, warp_cv2, (cols, rows))

	## TF version
	mid = np.array([moving.shape[1] / 2, moving.shape[0] / 2])
	x, y = np.sum(warp_cv2[0:2, 0:2] * mid, axis=1) + warp_cv2[:, 2] - mid
	theta = 180 * math.atan2(warp_matrix[0, 1], warp_matrix[0, 0]) / math.pi
	xytheta_TF = np.array([[-x, -y, theta]])
	batch = np.zeros((1, 265, 257, 2))
	batch[0, :, :, :] = np.stack((fixed, moving), axis=2)
	moved, xytheta0, theta0, cost = netR.transform_batch(batch, xytheta_TF)
	node1_transformed = moved[0].squeeze()
	cc = np.dot(node1_transformed.flatten(), fixed.flatten()) / np.linalg.norm(
		node1_transformed.flatten()) / np.linalg.norm(fixed.flatten())

	# combined = np.zeros(shape=(node1_transformed.shape[0],node1_transformed.shape[1],3))
	# combined[:,:,0]=node1_transformed/max(node1_transformed.flatten())
	# combined[:,:,1]=fixed
	# combined[:,:,2]=moving
	# plt.imshow(combined)
	# plt.show()
	# self.edges[edgeID].view(img2 = node1_transformed)
	return (cc, warp_cv2, xytheta_TF, node1_transformed)


# SITK non rigid reg
def deformable_reg(fixed, moving):
	fixed1 = sitk.GetImageFromArray(fixed, sitk.sitkFloat32)
	moving1 = sitk.GetImageFromArray(moving, sitk.sitkFloat32)
	demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
	demons.SetNumberOfIterations(2000)
	demons.SetStandardDeviations(5.0)

	warpField = demons.Execute(fixed1, moving1)
	outTx = sitk.DisplacementFieldTransform(warpField)

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed1)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0.)
	resampler.SetTransform(outTx)
	moved = sitk.GetArrayFromImage(resampler.Execute(moving1))

	E = get_jacobian_energy(outTx)
	cc = ecc(fixed, moved)
	return moved, cc, E


def deformable_reg_batch(batch):
	cc = [None] * batch.shape[0]
	E = [None] * batch.shape[0]
	for i in range(batch.shape[0]):
		batch[i, :, :, 1], cc[i], E[i] = deformable_reg(batch[i, :, :, 0], batch[i, :, :, 1])
	return batch, cc, E


def get_jacobian_energy(outTx):
	Sxy = sitk.GetArrayFromImage(outTx.GetDisplacementField())
	gx_y, gx_x = np.gradient(Sxy[:, :, 0])
	gy_y, gy_x = np.gradient(Sxy[:, :, 1])
	gx_x += 1
	gy_y += 1
	det_j = gx_x * gy_y - gy_x * gx_y
	return sum(np.square(det_j).flatten()) / (Sxy.shape[0] * Sxy.shape[1])


# misc affine matrix calculations
def affine_2d_to_3d(affine):
	new = np.eye(4)
	new[0:2, 0:2] = affine[0:2, 0:2]
	new[0:2, 3] = affine[0:2, 2]
	return new


def affine_make_full(affine):
	new = np.eye(max(affine.shape))
	new[0:affine.shape[0], 0:affine.shape[1]] = affine
	return new


def affine2d_from_xytheta(xytheta):
	x, y, theta = xytheta
	affine = np.array([[math.cos(math.radians(theta)), math.sin(math.radians(theta)), x],
					   [-1. * math.sin(math.radians(theta)), math.cos(math.radians(theta)), y],
					   [0, 0, 1]])
	return affine


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6


def affine_get_params(R):
	assert (isRotationMatrix(R[0:3, 0:3]))

	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

	singular = sy < 1e-6

	if not singular:
		theta_x = math.degrees(math.atan2(R[2, 1], R[2, 2]))
		theta_y = math.degrees(math.atan2(-R[2, 0], sy))
		theta_z = math.degrees(math.atan2(R[1, 0], R[0, 0]))
	else:
		theta_x = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
		theta_y = math.degrees(math.atan2(-R[2, 0], sy))
		theta_z = 0

	y = R[0, -1]
	x = R[1, -1]
	z = R[2, -1]

	return [theta_x, theta_y, theta_z, x, y, z]


def reverse_tform_order(dx, dy, theta):
	# calculate rotate -> translate from translate -> rotate
	dx1 = np.cos(np.deg2rad(theta)) * dx - np.sin(np.deg2rad(theta)) * dy
	dy1 = np.sin(np.deg2rad(theta)) * dx + np.cos(np.deg2rad(theta)) * dy

	# calculate translate -> rotate from rotate -> translate
	dx2 = (dx + (np.sin(np.deg2rad(theta)) / np.cos(np.deg2rad(theta))) * dy) / (
				np.cos(np.deg2rad(theta)) + ((np.sin(np.deg2rad(theta)) ** 2) / np.cos(np.deg2rad(theta))))
	dy2 = (dy - (np.sin(np.deg2rad(theta)) / np.cos(np.deg2rad(theta))) * dx) / (
				np.cos(np.deg2rad(theta)) + ((np.sin(np.deg2rad(theta)) ** 2) / np.cos(np.deg2rad(theta))))

	return [[dx1, dy1], [dx2, dy2]]

def fit_laplace(x, y):
	ul = x[np.argmax(y)]
	error = np.inf
	scale = 1
	while True:
		y_hat = stats.laplace.pdf(x, loc=ul, scale=scale)
		y_hat = np.max(y) * y_hat / np.max(y_hat)
		e = np.sum(np.square(y - y_hat))
		if e < error:
			error = e
			scale += 0.5
		else:
			sl = scale - 0.5
			break
	return ul, sl, np.max(y)/stats.laplace.pdf(ul,loc=ul,scale=sl)


def fit_pr_hypsecant(x, y, pow):
	ul = x[np.argmax(y)]
	error = np.inf
	scale = 0.1
	while True:
		y_hat = np.power(stats.hypsecant.pdf(x, loc=ul, scale=scale), pow)
		y_hat = np.max(y) * y_hat / np.max(y_hat)
		e = np.sum(np.square(y - y_hat))
		if e < error:
			error = e
			scale += 0.1
		else:
			sl = scale - 0.1
			break
	return ul, sl, np.max(y)/np.power(stats.hypsecant.pdf(ul, loc=ul, scale=sl), pow)