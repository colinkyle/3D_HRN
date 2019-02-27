import copy
import cv2
import json
import matplotlib.pyplot as plt
import math
import nibabel as nib
import numpy as np
import pickle
import SimpleITK as sitk
from scipy import misc
from scipy.interpolate import interpn
from scipy import stats
from TF_models import RigidNet, TpsNet
import tifffile
import fsys
import tensorflow as tf
import tqdm
import re


# TODO

# main image class
class Img(np.ndarray):
	"""
	Data type for 2D histology section data
	"""
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
		"""
		read image from disk, normalize intensities to [0,1], return as instance of Img
		:param fpath: path to image file
		:param metadata: optional metadata to associate with file
		:return: instance of Img
		"""
		[path, name, ext] = fsys.file_parts(fpath)
		if ext == '.tiff':
			# read META data
			with tifffile.TiffFile(fpath) as tif:
				data = tif.asarray()
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
			data = data / 255.0
			return cls.from_array(data, get_defaultMETA())

	@classmethod
	def from_array(cls, imgdat, metadata={}):
		"""
		create instance of Img from a matrix
		:param imgdat: image data
		:param metadata: optional metadata
		:return: instance of Img
		"""
		if not metadata:
			metadata = get_defaultMETA()
		return cls(
			shape=imgdat.shape,
			buffer=imgdat.astype(np.float32),
			dtype=np.float32,
			metadata=metadata)

	def imwrite(self, fpath):
		"""
		save image to disk as tiff
		:param fpath: save path (must have tiff extension)
		:return: none
		"""
		metadata = json.dumps(self.meta)
		tifffile.imsave(fpath, self, description=metadata)

	def view(self):
		"""
		view instance of Img
		:return: matplotlib plot
		"""
		plt.imshow(np.ndarray(
			shape=self.shape,
			buffer=self.astype(np.float32),
			dtype=np.float32), cmap='gray', interpolation='bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()

# defines relationship between images in terms of
# probability distributions of relative positions
class Edge:
	"""
	Edge class links 2 instances of Img with a probability distribution:
	p(dx), p(dy), p(dtheta), these are image similarity as a function of
	relative position between the images
	"""
	def __init__(self, img0, img1):
		"""
		set img0 as node0, img1 as node1, compute dz, and initialize
		x, y, and theta distribution parameters
		:param img0: instance of Img
		:param img1: instance of Img
		"""
		if (not isinstance(img0, Img)) or (not isinstance(img1, Img)):
			raise TypeError("Edge must reference two Img objects")
		self.node0 = img0
		self.node1 = img1

		self.dz = img1.meta['MAT0'][2][3] - img0.meta['MAT0'][2][3]

		self.param_x = [0, 6, 1] # position, scale, normalization factor
		self.param_y = [0, 6, 1] # position, scale, normalization factor
		self.param_theta = [0, 6, 1] # position, scale, normalization factor

	def p_x(self, dx):
		"""
		return value of lapace distribution at dx, with params: param_x
		:param dx: relative position between sections
		:return: p(x) within [0,1]
		"""
		px = self.param_x[2]*stats.laplace.pdf(dx, loc=self.param_x[0], scale=self.param_x[1])
		return px

	def p_y(self, dy):
		"""
		return value of power-raised hypersecant distribution at dy, with params: param_y
		power is raised to hardcoded 0.1 for underfit.
		See Faliva and Zoia, 2017, Entropy  doi:10.3390/e19040149
		:param dy: relative position between sections
		:return: p(y) within [0,1]
		"""
		py = self.param_y[2]*np.power(stats.hypsecant.pdf(dy, loc=self.param_y[0], scale=self.param_y[1]), 0.1)
		return py

	def p_theta(self, dtheta):
		"""
		return value of power-raised hypersecant distribution at dtheta, with params: param_theta
		power is raised to hardcoded 0.01 for an accurate fit.
		See Faliva and Zoia, 2017, Entropy  doi:10.3390/e19040149
		:param dtheta: relative angle between sections
		:return: p(theta) within [0,1]
		"""
		ptheta = self.param_theta[2] * np.power(stats.hypsecant.pdf(dtheta, loc=self.param_theta[0], scale=self.param_theta[1]), 0.01)
		return ptheta

	def view(self, img0=None, img1=None, img2=None):
		"""
		view pair of images in instance of Edge
		:return: matplotlib plot
		"""
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
class Stack:
	"""
	Stack is a container class that collects instances of Img,
	their file names, and a dictionary of Edges callable with
	tuple of image indices (i.e., Stack.edge[(0,1)]

	Stack represents an entire image stack and the relationships
	between neighboring images
	"""
	def __init__(self, fList=[], metaList=[]):
		"""
		Stack is container of Img objects, fnames, and edge dictionary
		:param fList: list of filename strings, to be loaded into Stack
		:param metaList: list of metadata for each Img
		edge network is initialzed with edge net of immediate neighbors
		"""
		self.fList = fList
		self.images = []
		self.edges = {}

		if self.fList:
			for i, fname in enumerate(self.fList):
				self.images.append(Img.imread(fname))
				self.images[-1].meta['fname'] = fname
		if metaList:
			self.update_meta(metaList)

		self.build_edge_net(1)

	def save(self, fname):
		"""
		saves as pickled object
		:param fname: filename to save to
		:return: none
		"""
		save_obj(self, fname)

	def update_meta(self, metaList):
		if len(metaList) == len(self.images):
			for i, meta in enumerate(metaList):
				self.images[i].meta.update(meta)
		else:
			raise ValueError('metaList must be same length as self.images')

	def build_edge_net(self, n_neighbors):
		"""
		builds edge network but does not estimate
		probability distributions between edges
		:param n_neighbors: how many closest neighbors to build edges between
		:return: updates edges attribute
		"""
		for i in range(len(self.images) - n_neighbors):
			for j in np.arange(i + 1, i + n_neighbors + 1):
				self.edges[(i, j)] = Edge(self.images[i], self.images[j])

	def estimate_edge_distributions(self, net_params):
		"""
		Fits parameters for p(dx), p(dy), p(dtheta) between histology secions in Edge.
		:param params: param dictionary to use to load TF_models.RigidNet
		:return: update edge attribute with fitted distribution parameters
		"""
		netR = RigidNet(net_params)
		# Loop edges

		for edge_id, edge in self.edges.items():
			print(edge_id)
			# Search for best registration in cv2, warp with STN
			(cc, warp_cv2, xytheta_TF, moved) = self.rigid_reg(edge_id,netR)

			# tf.reset_default_graph()
			# del netR

			## Compute empirical cc distribution within x,y,theta grid
			fixed, moving = self.edges[edge_id].node0, self.edges[edge_id].node1
			grid_size = 15
			batch_size = 75
			Xform = np.zeros((grid_size ** 3, 3))
			cnt = 0
			r = [30, 30, 10]
			num = 31
			#construct grid
			for i in np.linspace(-r[0], r[0], grid_size):
				for j in np.linspace(-r[1], r[1], grid_size):
					for k in np.linspace(-r[2], r[2], grid_size):
						Xform[cnt] = [i+xytheta_TF[0,0], j+xytheta_TF[0,1], k+xytheta_TF[0,2]]
						cnt += 1
			#batch of fixed movings with which to apply different xforms
			batch = np.zeros((batch_size, fixed.shape[0], fixed.shape[1], 2))
			for i in range(batch_size):
				batch[i] = np.stack((fixed, moving), axis=2)

			CC = np.zeros((grid_size, grid_size, grid_size))
			y, x, z = np.meshgrid(range(grid_size), range(grid_size), range(grid_size))
			y = y.flatten()
			x = x.flatten()
			z = z.flatten()
			#loop until all x,y,theta combos have been completed
			for b in range(int((grid_size**3)/batch_size)):
				#apply next subgrid of transforms
				tformed, cost_cc = netR.transform_with_xytheta(batch, Xform[b * batch_size:b * batch_size + batch_size])
				X, Y, Z = x[b * batch_size:b * batch_size + batch_size], y[b * batch_size:b * batch_size + batch_size], z[b * batch_size:b * batch_size + batch_size]
				# record cc values
				for i in range(tformed.shape[0]):
					CC[X[i], Y[i], Z[i]] = ecc(fixed, tformed[i])

			#reduce grid to marginal max values
			dist_x = np.max(np.max(CC, axis=2, keepdims=True), axis=1, keepdims=True).flatten()
			dist_y = np.max(np.max(CC, axis=2, keepdims=True), axis=0, keepdims=True).flatten()
			dist_theta = np.max(np.max(CC, axis=1, keepdims=True), axis=0, keepdims=True).flatten()
			# add best cc found by cv2
			dist_x = np.concatenate((dist_x, [cc]), axis=0)
			dist_y = np.concatenate((dist_y, [cc]), axis=0)
			dist_theta = np.concatenate((dist_theta, [cc]), axis=0)

			x = np.linspace(-r[0], r[0], grid_size)
			y = np.linspace(-r[1], r[1], grid_size)
			z = np.linspace(-r[2], r[2], grid_size)

			# add location of best cc found by cv2
			x = np.concatenate([x, [0]], axis=0)
			y = np.concatenate([y, [0]], axis=0)
			z = np.concatenate([z, [0]], axis=0)
			# add location of best registration in case not found in grid
			#shift values so they are relative to no-transformed images
			# rather than being relative to pre-registered moving image
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

			#finally fit and update edge distribution parameters
			edge.param_x = fit_laplace(x, dist_x)
			edge.param_y = fit_pr_hypsecant(y, dist_y, .1)
			edge.param_theta = fit_pr_hypsecant(z, dist_theta, .01)

	def rigid_reg(self, edgeID,netR):
		"""
		performs rigid registration in CV2 in case network
		wont find best params
		:param edgeID: tuple that IDs edge
		:param netR: rigid network to transform images with
		:return:
		cc: max similarity found
		warp_cv2: cv2 warp params (corner centered)
		xytheta_TF: network warp params (middle centered)
		moved: resampled moving image
		"""
		(cc, warp_cv2, xytheta_TF, moved) = rigid_reg(self.edges[edgeID].node0, self.edges[edgeID].node1,netR)
		return (cc, warp_cv2, xytheta_TF, moved)

	def score_param_consistency(self, xytheta):
		"""
		calculates probability of stack warps to template given
		stack-wize image similarity structure
		:param xytheta: list of transforms each image in stack -> template
		:return: p(template_position|xytheta,stack)
		"""
		total_err = 0
		#loop edges
		for pair in self.edges.keys():
			# center rotated warps from origin to template
			warp1 = affine2d_from_xytheta(xytheta[pair[0]])  # img1 warp
			warp2 = affine2d_from_xytheta(xytheta[pair[1]])  # img2 warp

			# calculate img1 -> img0 relative positions of
			# neighbors after warping to template
			warp3 = np.dot(warp1, np.linalg.inv(warp2))
			_, _, d_tz, d_x, d_y, _ = affine_get_params(affine_2d_to_3d(warp3))

			# get probabilities from edge distributions
			p_x = self.edges[pair].p_x(d_x)
			p_y = self.edges[pair].p_y(d_y)
			p_theta = self.edges[pair].p_theta(d_tz)
			# cost function
			p_edge = p_x * p_y * p_theta# * (1 - .05 ** np.diff(pair))

			if not np.isnan(p_edge):
				total_err += np.log(p_edge)
		return np.exp(total_err / len(self.edges.keys()))

	def set_warped_from_moved(self,moved,xytheta=None,splines=None):
		"""
		Function sets Img.warped (convenient to store Original and
		Warped image together sometimes)
		:param moved: resampled moving images
		:param xytheta: params to store (convert original to resampled)
		:param splines: params to store (convert rigid resampled to final resampled)
		:return:
		updates Img attributes
		"""
		if len(self.images) != moved.shape[0]:
			raise ValueError('tformed.shape[0] must be same length as self.images')
		for i in range(len(self.images)):
			self.images[i].warped = moved[i].squeeze()
		if xytheta is not None:
			for i in range(len(self.images)):
				self.images[i].xytheta = xytheta[i]
		if splines is not None:
			for i in range(len(self.images)):
				self.images[i].splines = splines[i]

	def get_nifti3D_from_warped_stack(self):
		"""
		returns a 3D nifti object from a warped stack
		"""
		locs = [int(re.search('[0-9]*[N]', self.fList[i]).group(0)[0:-1]) for i in range(len(self.fList)) if
				re.search('[0-9]*[N]', self.fList[i])]
		#z_middle = (max(locs) + min(locs)) / 2.

		z_all = np.arange(min(locs), max(locs), 1)
		z_arg = np.empty(z_all.shape)
		z_arg[:] = None

		for i, f in enumerate(self.fList):
			z_pos = int(re.search('[0-9]*[N]', f).group(0)[0:-1])
			arg_loc = np.argwhere(z_all == z_pos)
			z_arg[arg_loc] = i

		hist_data = np.zeros((self.images[0].shape[0], self.images[0].shape[1], len(z_arg)))
		for z in range(len(z_arg)):
			if ~np.isnan(z_arg[z]):
				hist_data[:, :, z] = self.images[np.int(z_arg[z])].warped

		affine = np.array(copy.deepcopy(self.images[0].meta['MAT']))
		affine[2, 3] = -affine[2, 2] * ((max(locs) - min(locs)) / 2.)
		return nib.Nifti1Image(hist_data, affine)

class Template:
	"""
	MRI container class for the 3D template image
	"""
	def __init__(self, fname):
		"""
		loads data and metadata of MRI image
		:param fname: filename of MRI image
		"""
		self.affine = nib.load(fname).affine
		self.data = nib.load(fname).get_data()
		self.data = self.data / max(self.data.flatten())

	def retrain_TF_R(self, net, big_batch, ntrain=100, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, moved = net.train_unsupervised(batch)

			#print('count: {}, cost: {}, sanity: {}'.format(cnt, cost,np.mean(np.abs(moved))))
		return net

	def retrain_TF_E(self, net, big_batch, ntrain=500, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, cost_cc, E_det_j, E_div, E_curl = net.train(batch)

			# print('count: {}, cost_cc: {}, sanity: {}'.format(cnt, cost_t, np.mean(np.abs(out), axis=0)))
		return net

	def retrain_TF_Both(self, netR, netE, big_batch, ntrain=500, nbatch=10):
		print('\nretraining model...')
		## estimate transformations
		for _ in tqdm.tqdm(range(ntrain)):
			batch = big_batch[np.random.choice(range(big_batch.shape[0]), nbatch), :, :, :]
			cnt, cost, moved = netR.train_unsupervised(batch)
			#print('count_R: {}, cost_mi: {}'.format(cnt, cost))
			for i in range(nbatch):
				batch[i, :, :, 1] = np.squeeze(moved[i, :, :])
			cnt, cost, cost_cc, E_det_j, E_div, E_curl = netE.train(batch)

			#print('count_E: {}, cost_mi: {}'.format(cnt, cost))
		return netR, netE

	def gen_batch(self, IMG, z_loc, theta_x, theta_y, dxy, dz, subsample=False, n_subsample=0):
		"""
		Generates a batch of template-section,histology pairs for use with STN
		:param IMG: Stack histology stack
		:param z_loc: z transform to template
		:param theta_x: theta_x transform to template
		:param theta_y: theta_y transform to template
		:param dxy: xy scaling transform to template
		:param dz: z scaling transform to template
		:param subsample: whether or not to subsample for training batch at many transform params
		:param n_subsample: how many subsamples to return
		:return: batch matrix shape: (len(Stack)-or-n_subsample, 265, 257, 2)
		"""
		#generate affine fore resampling MRI at theta_x, theta_y, dxy, dz, and z locations of each section
		# set xyz resolution
		mri_affine = copy.deepcopy(self.affine)
		mri_affine[:2, :] = dxy * mri_affine[:2, :]
		mri_affine[2, :] = dz * mri_affine[2, :]
		# x,y,z MRI
		x_mri = np.array([mri_affine[0][0] * x + mri_affine[0][3] for x in range(self.data.shape[0])])
		y_mri = np.array([mri_affine[1][1] * y + mri_affine[1][3] for y in range(self.data.shape[1])])
		z_mri = np.array([mri_affine[2][2] * z + mri_affine[2][3] for z in range(self.data.shape[2])])

		#theta_x rotation
		affine_x = np.array([[1, 0, 0, 0],
							 [0, math.cos(math.radians(theta_x)), -1. * math.sin(math.radians(theta_x)), 0],
							 [0, math.sin(math.radians(theta_x)), math.cos(math.radians(theta_x)), 0],
							 [0, 0, 0, 1]])
		#theta y rotation
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
				# pixel locations x,y,z of each section
				img = IMG.images[samp]
				x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
				y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
				# add z shift
				z = img.meta['MAT'][2][3] + z_loc
				XX, YY = np.atleast_2d(x, y)
				YY = YY.T
				# warp according to MRI warp params
				XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
				YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
				ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
				# interp_func = RegularGridInterpolator((x_mri, y_mri, z_mri),self.data,bounds_error=False,
				#             fill_value=0)
				# mri = Img.from_array(
				#     interp_func(np.array([XXX, YYY, ZZZ]).T))
				# resample intensities at section locations
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
			# Match img to mri
			# pixel locations x,y,z of each section
			x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
			y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
			# add z shift
			z = img.meta['MAT'][2][3] + z_loc
			XX, YY = np.atleast_2d(x, y)
			YY = YY.T
			# warp according to MRI warp params
			XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
			YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
			ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
			# interpolate resampled template intensities
			mri = Img.from_array(
				interpn((x_mri, y_mri, z_mri), self.data, np.array([XXX, YYY, ZZZ]).T,
						bounds_error=False, fill_value=0))

			# histogram matching of intensities
			matcher = sitk.HistogramMatchingImageFilter()
			matcher.SetNumberOfHistogramLevels(512)
			matcher.SetNumberOfMatchPoints(30)
			img = matcher.Execute(sitk.GetImageFromArray(img, sitk.sitkFloat32),
								  sitk.GetImageFromArray(mri, sitk.sitkFloat32))
			img = sitk.GetArrayFromImage(img)
			batch[i, :, :, :] = cv2.merge((mri, img))
		return batch

	def get_nifti3D_from_warped_template(self, IMG, z_loc, theta_x, theta_y, dxy, dz):
		"""
		returns a 3D nifti object of warped template image
		"""
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

		new_mri = np.zeros((IMG.images[0].shape[0], IMG.images[0].shape[1], self.data.shape[2]))

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

def ecc(img0, img1):
	return np.dot(img0.flatten(), img1.flatten()) / np.linalg.norm(img0.flatten()) / np.linalg.norm(img1.flatten())

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
	batch = np.zeros((1, fixed.shape[0], fixed.shape[1], 2))
	batch[0, :, :, :] = np.stack((fixed, moving), axis=2)
	moved, cost = netR.transform_with_xytheta(batch, xytheta_TF)
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

# misc affine matrix calculations
def affine_2d_to_3d(affine):
	new = np.eye(4)
	new[0:2, 0:2] = affine[0:2, 0:2]
	new[0:2, 3] = affine[0:2, 2]
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