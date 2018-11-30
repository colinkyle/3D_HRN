from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math


class RigidNet:
	"""
	CNN and rigid Spatial Transformer Network
	"""
	def __init__(self, params):
		"""
		Create Network:
		Major Components:
		1. Layer 1:
			Convolutional: 15pixel by 15pixel by
				2 channel (fixed image, moving image) by
				24 filters
			Stride: 5
			Activation Function: ReLU
		2. Layer 2:
			Convolutional: 15 by 15 by
				24 channel (filters layer 1) by
				52 filters
			Stride: 1
			Activation Function: ReLU
		3. --Dropout--
		4. Layer 3:
			Fully Connected: 128 outputs
			Activation Function: ReLU
		5. Layer 4:
			Fully Connected: 3 outputs (x,y,theta)
			Activation Function: None/Linear
		6. Cost function on squared error of rigid params
				(for supervised learning on output parameters)
		7. RMSprop (Supervised)
		8. Rigid STN
		9. Cost function (CC, Squared Error) Best results with
				Squared Error for unsupervised learning on
				fixed vs. moved intensities

		:param params:
		params = {'load_file': 'D:/Absolute/Path/To/ModelName',
		  'save_file': 'RigidNetUniqueID',
		  'save_interval': 1000,
		  'batch_size': 32,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.999,  # RMS Prop decay
		  'rms_eps': 1e-9,  # RMS Prop epsilon
		  'width': 265, # (confusing- vertical dimension of histology images)
		  'height': 257,# (confusing- horizontal dimension of histology images)
		  'numParam': 3,# output to STN x,y,theta for resampling the moving image
		  'train': True, # Enables/Disables dropout
		  }
		"""
		g_rigid = tf.Graph()
		with g_rigid.as_default():
			with tf.device('/gpu:0'):
				"""params has width, depth, numParam"""
				self.params = params
				self.network_name = 'AtlasNet'
				# self.sess = tf.Session()
				config = tf.ConfigProto(allow_soft_placement=True)
				self.sess = tf.Session(config=config)
				# Runs the op.
				# print(self.sess.run(c))

				self.In = tf.placeholder('float', [None, params['width'], params['height'], 2],
										 name='In')
				self.Ideal = tf.placeholder('float', [None, params['numParam']], name=self.network_name + '_Ideal')

				# Layer 1 (Convolutional)
				layer_name = 'conv1'
				size = 15
				channels = 2
				filters = 24
				stride = 5
				self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.c1 = tf.nn.conv2d(self.In, self.w1, strides=[1, stride, stride, 1], padding='SAME',
									   name=self.network_name + '_' + layer_name + '_convs')
				self.o1 = tf.nn.relu(tf.add(self.c1, self.b1),
									 name=self.network_name + '_' + layer_name + '_activations')

				# Layer 2 (Convolutional)
				layer_name = 'conv2'
				size = 15
				channels = 24
				filters = 52
				stride = 1
				self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',
									   name=self.network_name + '_' + layer_name + '_convs')
				self.o2 = tf.nn.relu(tf.add(self.c2, self.b2),
									 name=self.network_name + '_' + layer_name + '_activations')

				o2_shape = self.o2.get_shape().as_list()

				# Layer 3 (Fully connected)
				layer_name = 'fc4'
				hiddens = 128
				dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
				self.o2_flat = tf.reshape(self.o2, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
				# dropout
				if params['train']:
					self.keep_prob = tf.placeholder(tf.float32)
					self.o2_flat = tf.nn.dropout(self.o2_flat, self.keep_prob)

				self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.ip4 = tf.add(tf.matmul(self.o2_flat, self.w4), self.b4,
								  name=self.network_name + '_' + layer_name + '_ips')
				self.o4 = tf.nn.relu(self.ip4, name=self.network_name + '_' + layer_name + '_activations')

				# Layer 4 (Fully connected, parameter output)
				layer_name = 'fc5'
				hiddens = params['numParam']
				dim = 128
				self.w5 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.xytheta = tf.add(tf.matmul(self.o4, self.w5), self.b5, name='Out')

				#### Cost,Optimizer train on parameters (supervised) ####
				self.cost_sup = tf.reduce_sum(tf.pow(tf.subtract(self.xytheta, self.Ideal), 2))

				if self.params['load_file'] is not None:
					try:
						self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step',
												   trainable=False)
					except:
						self.global_step = tf.Variable(0, name='global_step', trainable=False)
				else:
					self.global_step = tf.Variable(0, name='global_step', trainable=False)

				# Gradient descent on loss function
				self.rmsprop_sup = tf.train.RMSPropOptimizer(self.params['lr'],
															 epsilon=self.params['rms_eps']).minimize(self.cost_sup,
																									  global_step=self.global_step)
				#### Do resampling of moving image ####
				self.moving = tf.slice(self.In, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
				self.tformed = self.transformer([params['width'], params['height']])

				fixed = tf.slice(self.In, [0, 0, 0, 0],
								 [-1, params['width'], params['height'], 1])  # ,(-1, params['width']*params['height'])

				#### Cost,Optimizer train on similarity function (unsupervised) ####
				self.cost_cc = ncc(fixed, self.tformed)
				self.cost_sq_er = tf.reduce_sum(tf.pow(tf.subtract(self.tformed, fixed), 2), name='RMS')

				self.rmsprop_unsup = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(
					self.cost_sq_er,
					global_step=self.global_step)
				self.saver = tf.train.Saver(max_to_keep=0)

				self.sess.run(tf.global_variables_initializer())

				if self.params['load_file'] is not None:
					print('Loading checkpoint...')
					self.saver.restore(self.sess, self.params['load_file'])

				momentum_initializers = [var.initializer for var in tf.global_variables() if 'RMS' in var.name]
				self.sess.run(momentum_initializers)

	def train_unsupervised(self, batch):
		"""
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:return:
		training_count:
		 shape=1
		cc: image similarity
		 shape = (batch_size)
		"""
		feed_dict = {self.In: batch, self.keep_prob: .60}
		_, cnt, cc = self.sess.run(
			[self.rmsprop_unsup, self.global_step, self.cost_cc],
			feed_dict=feed_dict)
		return cnt, cc

	def train_supervised(self, batch, ideal_xytheta):
		"""
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:param ideal_xytheta: known parameters that would results in aligned images
			shape = (batch_size, 3)
		:return:
		training_count:
		 shape=1
		squared error: (model_xytheta - ideal_xytheta)**2
		cc: image similarity
		 shape = (batch_size)
		"""
		feed_dict = {self.In: batch, self.Ideal: ideal_xytheta, self.keep_prob: .60}
		_, cnt, cost, cc = self.sess.run(
			[self.rmsprop_sup, self.global_step, self.cost_sup, self.cost_cc], feed_dict=feed_dict)
		# print(self.sess.run([self.rmsprop_sup, self.global_step, self.cost_sup, self.xytheta, self.cost_cc], feed_dict=feed_dict))
		return cnt, cost / batch.shape[0], cc

	def save_ckpt(self, filename):
		"""
		saves model
		:param filename: filename string
		:return: None
		"""
		self.saver.save(self.sess, filename)

	def run(self, batch):
		"""
		Registers a batch of images
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:return:
		moved images:
		 shape = (batch_size, image_width, image_height)
		xytheta: params used to warp images
		 shape = (batch_size,3)
		cost_cc: similarity
		 shape = (batch_size)
		"""
		feed_dict = {self.In: batch, self.keep_prob: 1}
		moved, xytheta, cost_cc = self.sess.run([self.tformed, self.xytheta, self.cost_cc], feed_dict=feed_dict)

		return moved, xytheta, cost_cc

	def transform_with_xytheta(self, batch, xytheta):
		"""
		transforms a batch of images according to xytheta
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:param xytheta: parameters used to transform moving images
			shape = (batch_size, 3)
		:return:
		moved images:
		 shape = (batch_size, image_width, image_height)
		cost_cc: similarity
		 shape = (batch_size)
		"""
		feed_dict = {self.In: batch, self.xytheta: xytheta}
		moved, cost_cc = self.sess.run([self.tformed, self.cost_cc],
														  feed_dict=feed_dict)

		return moved, cost_cc

	def transformer(self, out_size, name='SpatialTransformer', **kwargs):
		"""Spatial Transformer Layer

		Implements a spatial transformer layer as described in [1]_.
		Based on [2]_ and edited by David Dao for Tensorflow.

		Parameters
		----------
		U : float
			The output of a convolutional net should have the
			shape [num_batch, height, width, num_channels].
		theta: float
			The output of the
			localisation network should be [num_batch, 6].
		out_size: tuple of two ints
			The size of the output of the network (height, width)

		References
		----------
		.. [1]  Spatial Transformer Networks
				Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
				Submitted on 5 Jun 2015
		.. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

		Notes
		-----
		To initialize the network to the identity transform init
		``theta`` to :
			identity = np.array([[1., 0., 0.],
								 [0., 1., 0.]])
			identity = identity.flatten()
			theta = tf.Variable(initial_value=identity)

		"""

		def _repeat(x, n_repeats):
			with tf.variable_scope('_repeat'):
				rep = tf.transpose(
					tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
				rep = tf.cast(rep, 'int32')
				x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
				return tf.reshape(x, [-1])

		def _interpolate(im, x, y, out_size):
			with tf.variable_scope('_interpolate'):
				# constants
				num_batch = tf.shape(im)[0]
				height = tf.shape(im)[1]
				width = tf.shape(im)[2]
				channels = tf.shape(im)[3]

				x = tf.cast(x, 'float32')
				y = tf.cast(y, 'float32')
				height_f = tf.cast(height, 'float32')
				width_f = tf.cast(width, 'float32')
				out_height = out_size[0]
				out_width = out_size[1]
				zero = tf.zeros([], dtype='int32')
				max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
				max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

				# scale indices from [-1, 1] to [0, width/height]
				x = (x + 1.0) * (width_f) / 2.0
				y = (y + 1.0) * (height_f) / 2.0

				# do sampling
				x0 = tf.cast(tf.floor(x), 'int32')
				x1 = x0 + 1
				y0 = tf.cast(tf.floor(y), 'int32')
				y1 = y0 + 1

				x0 = tf.clip_by_value(x0, zero, max_x)
				x1 = tf.clip_by_value(x1, zero, max_x)
				y0 = tf.clip_by_value(y0, zero, max_y)
				y1 = tf.clip_by_value(y1, zero, max_y)
				dim2 = width
				dim1 = width * height
				base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
				base_y0 = base + y0 * dim2
				base_y1 = base + y1 * dim2
				idx_a = base_y0 + x0
				idx_b = base_y1 + x0
				idx_c = base_y0 + x1
				idx_d = base_y1 + x1

				# use indices to lookup pixels in the flat image and restore
				# channels dim
				im_flat = tf.reshape(im, tf.stack([-1, channels]))
				im_flat = tf.cast(im_flat, 'float32')
				Ia = tf.gather(im_flat, idx_a)
				Ib = tf.gather(im_flat, idx_b)
				Ic = tf.gather(im_flat, idx_c)
				Id = tf.gather(im_flat, idx_d)

				# and finally calculate interpolated values
				x0_f = tf.cast(x0, 'float32')
				x1_f = tf.cast(x1, 'float32')
				y0_f = tf.cast(y0, 'float32')
				y1_f = tf.cast(y1, 'float32')
				wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
				wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
				wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
				wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
				output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
				return output

		def _meshgrid(height, width):
			with tf.variable_scope('_meshgrid'):
				# This should be equivalent to:
				#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
				#                         np.linspace(-1, 1, height))
				#  ones = np.ones(np.prod(x_t.shape))
				#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
				x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
								tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
				y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
								tf.ones(shape=tf.stack([1, width])))

				x_t_flat = tf.reshape(x_t, (1, -1))
				y_t_flat = tf.reshape(y_t, (1, -1))

				ones = tf.ones_like(x_t_flat)
				grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
				return grid

		def _transform(self, theta, input_dim, out_size):
			with tf.variable_scope('_transform'):
				num_batch = tf.shape(input_dim)[0]
				height = tf.shape(input_dim)[1]
				width = tf.shape(input_dim)[2]
				#num_channels = tf.shape(input_dim)[3]
				# theta = tf.reshape(theta, (-1, 2, 3))
				# theta = tf.cast(theta, 'float32')

				DX = tf.slice(theta, [0, 0], [-1, 1]) / (out_size[1] / 2)  # try swap
				DY = tf.slice(theta, [0, 1], [-1, 1]) / (out_size[0] / 2)
				DRot = (math.pi / 180) * tf.slice(theta, [0, 2], [-1, 1])

				theta2 = tf.concat([tf.cos(DRot), tf.sin(DRot), DX, -tf.sin(DRot), tf.cos(DRot), DY], axis=1)
				theta2 = tf.reshape(theta2, (-1, 2, 3))
				#self.theta2 = tf.cast(theta2, 'float32')

				# grid of (x_t, y_t, 1), eq (1) in ref [1]
				#height_f = tf.cast(height, 'float32')
				#width_f = tf.cast(width, 'float32')
				out_height = out_size[0]
				out_width = out_size[1]
				grid = _meshgrid(out_height, out_width)
				grid = tf.expand_dims(grid, 0)
				grid = tf.reshape(grid, [-1])
				grid = tf.tile(grid, tf.stack([num_batch]))
				self.grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

				# Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
				T_g = tf.matmul(theta2, self.grid)
				x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
				y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
				self.x_s_flat = tf.reshape(x_s, [-1])
				self.y_s_flat = tf.reshape(y_s, [-1])

				input_transformed = _interpolate(
					input_dim, self.x_s_flat, self.y_s_flat,
					out_size)

				output = tf.reshape(
					input_transformed, tf.stack([num_batch, out_height, out_width, 1]))
				return output

		with tf.variable_scope(name):
			output = _transform(self, self.xytheta, self.moving, out_size)
			return output

class TpsNet:
	"""
	CNN and thin-plate spline Spatial Transformer Network
	"""
	def __init__(self, params):
		"""Create Network:
		Major Components:
		1. Layer 1:
			Convolutional: 15pixel by 15pixel by
				2 channel (fixed image, moving image) by
				24 filters
			Stride: 5
			Activation Function: ReLU
		2. Layer 2:
			Convolutional: 15 by 15 by
				24 channel (filters layer 1) by
				52 filters
			Stride: 1
			Activation Function: ReLU
		3. --Dropout--
		4. Layer 3:
			Fully Connected: 128 outputs
			Activation Function: ReLU
		5. Layer 4:
			Fully Connected: 128 outputs (8x8 x, 8x8 y) default for tps stn
			Activation Function: None/Linear
		6. thin-plate spline STN with regularizer penalties:
			determinant of Jacobian, Divergence, Curl
		7. Cost functions (Squared Error, regularized squared error, cc)
				Best results with reqularized squared error
		8. RMSprop

		:param params:
		params = {'load_file': 'D:/Absolute/Path/To/ModelName',
		  'save_file': 'TPSNetUniqueID',
		  'save_interval': 1000,
		  'batch_size': 32,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.999,  # RMS Prop decay
		  'rms_eps': 1e-9,  # RMS Prop epsilon
		  'width': 265, # (confusing- vertical dimension of histology images)
		  'height': 257,# (confusing- horizontal dimension of histology images)
		  'numParam': 3,# output to STN x,y,theta for resampling the moving image
		  'train': True, # Enables/Disables dropout
		  }"""
		g_elastic = tf.Graph()
		with g_elastic.as_default():
			with tf.device('/gpu:0'):
				self.params = params
				self.network_name = 'ElasticNet'
				config = tf.ConfigProto(allow_soft_placement=True)
				self.sess = tf.Session(config=config)
				# input placeholder
				self.In = tf.placeholder(tf.float32, [None, params['width'], params['height'], 2])
				# ? labels ?
				# self.Ideal = tf.placeholder(tf.float32, [params['batch_size'], params['width'], params['height'], 2], name=self.network_name + '_Ideal')

				# now do network for parameter creation
				# Layer 1 (Convolutional)
				layer_name = 'conv1'
				size = 15
				channels = 2
				filters = 24
				stride = 5
				self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.c1 = tf.nn.conv2d(self.In, self.w1, strides=[1, stride, stride, 1], padding='SAME',
									   name=self.network_name + '_' + layer_name + '_convs')
				self.o1 = tf.nn.relu(tf.add(self.c1, self.b1),
									 name=self.network_name + '_' + layer_name + '_activations')

				# Layer 2 (Convolutional)
				layer_name = 'conv2'
				size = 15
				channels = 24
				filters = 52
				stride = 1
				self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',
									   name=self.network_name + '_' + layer_name + '_convs')
				self.o2 = tf.nn.relu(tf.add(self.c2, self.b2),
									 name=self.network_name + '_' + layer_name + '_activations')

				o2_shape = self.o2.get_shape().as_list()

				# Layer 3 (Fully connected)
				layer_name = 'fc3'
				hiddens = 128
				dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
				self.o2_flat = tf.reshape(self.o2, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')

				# dropout
				if params['train']:
					self.keep_prob = tf.placeholder(tf.float32)
					self.o2_flat = tf.nn.dropout(self.o2_flat, self.keep_prob)

				self.w3 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.ip3 = tf.add(tf.matmul(self.o2_flat, self.w3), self.b3,
								  name=self.network_name + '_' + layer_name + '_ips')
				self.o3 = tf.nn.relu(self.ip3, name=self.network_name + '_' + layer_name + '_activations')

				# Let the output size of the projective transformer be quarter of the image size.
				self.outsize = (int(params['width']), int(params['height']))

				# Elastic Transformation Layer
				self.stl = ElasticTransformer(self.outsize)

				# Layer 4 (Fully connected output)
				layer_name = 'fc4'
				# hiddens correspond to elastic grid control points
				hiddens = self.stl.param_dim
				dim = 128
				self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
									  name=self.network_name + '_' + layer_name + '_weights')
				self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),
									  name=self.network_name + '_' + layer_name + '_biases')
				self.Theta = tf.add(tf.matmul(self.o3, self.w4), self.b4, name='Out')

				self.result, self.E_det_j, self.E_div, self.E_curl = self.stl.transform(self.In, self.Theta)
				# Cost,Optimizer
				bat_size, wid, height = tf.shape(self.In)[0], tf.shape(self.In)[1], tf.shape(self.In)[2]
				bat_size = tf.cast(bat_size, dtype=tf.float32)
				wid = tf.cast(wid, dtype=tf.float32)
				height = tf.cast(height, dtype=tf.float32)
				self.weight_se = tf.placeholder(tf.float32, shape=[1])
				self.weight_E_det_j = tf.placeholder(tf.float32, shape=[1])
				self.weight_E_div = tf.placeholder(tf.float32, shape=[1])
				self.weight_E_curl = tf.placeholder(tf.float32, shape=[1])
				# squared intensity error
				self.cost_se = tf.reduce_mean(tf.pow(tf.subtract(self.result[:, :, :, 1], self.In[:, :, :, 0]), 2))
				# self.cost_e = tf.abs((tf.reduce_sum(self.Energy)/bat_size) - 1)
				self.cost = self.weight_se * self.cost_se + \
							self.weight_E_det_j * tf.reduce_mean(self.E_det_j) + \
							self.weight_E_div * tf.reduce_mean(self.E_div) + \
							self.weight_E_curl * tf.reduce_mean(self.E_curl)
				self.cost2 = self.weight_E_det_j * tf.reduce_mean(self.E_det_j) + \
							 self.weight_E_div * tf.reduce_mean(self.E_div) + \
							 self.weight_E_curl * tf.reduce_mean(self.E_curl)
				# self.cost = MI(self.result[:,:,:,1],self.In[:,:,:,0],[params['batch_size'], params['width'], params['height']],10)
				if self.params['load_file'] is not None:
					try:
						self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step',
												   trainable=False)
					except:
						self.global_step = tf.Variable(0, name='global_step', trainable=False)
				else:
					self.global_step = tf.Variable(0, name='global_step', trainable=False)

				# Gradient descent on loss function
				self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(
					self.cost,
					global_step=self.global_step)
				self.fixed = tf.slice(self.In, [0, 0, 0, 0], [-1, params['width'], params['height'], 1])
				self.moved = tf.slice(self.result, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
				self.cost_cc = ncc(self.fixed, self.moved)
				self.saver = tf.train.Saver(max_to_keep=0)

				self.sess.run(tf.global_variables_initializer())

				if self.params['load_file'] is not None:
					print('Loading checkpoint...')
					self.saver.restore(self.sess, self.params['load_file'])

	def train(self, batch):
		"""
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:return:
		training_count:
		 shape=1
		regularized cost:
		 shape = (batch_size)
		cc: image similarity
		 shape = (batch_size)
		E_det_j: determinant of jacobian
		E_div: Divergence
		E_curl: Curl
		"""
		feed_dict = {self.In: batch,
					 self.keep_prob: .60,
					 self.weight_se: [0.25],
					 self.weight_E_det_j: [0.5],
					 self.weight_E_div: [0.5],
					 self.weight_E_curl: [0.5]}
		_, cnt, cost, cost_cc, E_det_j, E_div, E_curl = self.sess.run(
			[self.rmsprop, self.global_step, self.cost, self.cost_cc, self.E_det_j, self.E_div, self.E_curl],
			feed_dict=feed_dict)
		return cnt, cost, cost_cc, E_det_j, E_div, E_curl

	def run(self, batch):
		"""
		Registers a batch of images
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:return:
		moved images:
		 shape = (batch_size, image_width, image_height)
		splines: params used to warp images
		 shape = (batch_size,128)
		cost_cc: similarity
		 shape = (batch_size)
		cost: regularized cost
		cost2: regularization terms only
		"""
		feed_dict = {self.In: batch,
					 self.keep_prob: 1,
					 self.weight_se: [1.0],
					 self.weight_E_det_j: [3.0],
					 self.weight_E_div: [3.0],
					 self.weight_E_curl: [3.0]}
		moved, splines, cc, cost, cost2 = self.sess.run(
			[self.moved, self.Theta, self.cost_cc, self.cost, self.cost2], feed_dict=feed_dict)
		return moved, splines, cc, cost, cost2

	def save_ckpt(self, filename):
		"""
		saves model
		:param filename: filename string
		:return: None
		"""
		self.saver.save(self.sess, filename)

	def transform_with_splines(self, batch, splines):
		"""
		transforms a batch of images according to splines
		:param batch: batch of fixed, moving pairs
			shape=(batch_size, image_width, image_height, 2)
				in last dimension: 1st entry: fixed image, 2nd entry: moving
		:param splines: parameters used to transform moving images
			shape = (batch_size, 128)
		:return:
		moved images:
		 shape = (batch_size, image_width, image_height)
		cost_cc: similarity
		 shape = (batch_size)
		"""
		feed_dict = {self.In: batch, self.Theta: splines}
		moved, cost_cc = self.sess.run([self.moved, self.cost_cc],
														  feed_dict=feed_dict)
		return moved, cost_cc


class ElasticTransformer(object):
	"""Spatial Elastic Transformer Layer with Thin Plate Spline deformations

	Implements a spatial transformer layer as described in [1]_.
	Based on [4]_ and [5]_. Edited by Daniyar Turmukhambetov.

	"""

	def __init__(self, out_size, param_dim=2 * 64, name='SpatialElasticTransformer', interp_method='bilinear',
				 **kwargs):
		"""
		Parameters
		----------
		out_size : tuple of two ints
			The size of the output of the spatial network (height, width).
		param_dim: int
			The 2 x number of control points that define
			Thin Plate Splines deformation field.
			number of control points *MUST* be a square of an integer.
			2 x 16 by default.
		name : string
			The scope name of the variables in this network.

		"""
		num_control_points = int(param_dim / 2)
		assert param_dim == 2 * num_control_points, 'param_dim must be 2 times a square of an integer.'

		self.name = name
		self.param_dim = param_dim
		self.interp_method = interp_method
		self.num_control_points = num_control_points
		self.out_size = out_size

		self.grid_size = math.floor(math.sqrt(self.num_control_points))
		assert self.grid_size * self.grid_size == self.num_control_points, 'num_control_points must be a square of an int'

		with tf.variable_scope(self.name):
			# Create source grid
			self.source_points = ElasticTransformer.get_meshgrid(self.grid_size, self.grid_size)
			# Construct pixel grid
			self.pixel_grid = ElasticTransformer.get_meshgrid(self.out_size[1], self.out_size[0])
			self.num_pixels = self.out_size[0] * self.out_size[1]
			self.pixel_distances, self.L_inv = self._initialize_tps(self.source_points, self.pixel_grid)

	def transform(self, inp, theta, forward=True, batch_size=32, **kwargs):
		"""
		Parameters
		----------
		inp : float
			The input tensor should have the shape
			[batch_size, height, width, num_channels].
		theta: float
			Should have the shape of [batch_size, self.num_control_points x 2]
			Theta is the output of the localisation network, so it is
			the x and y offsets of the destination coordinates
			of each of the control points.
		Notes
		-----
		To initialize the network to the identity transform initialize ``theta`` to zeros:
			identity = np.zeros(16*2)
			identity = identity.flatten()
			theta = tf.Variable(initial_value=identity)

		"""
		with tf.variable_scope(self.name):
			# reshape destination offsets to be (batch_size, 2, num_control_points)
			# and add to source_points
			self.dxy = tf.reshape(theta, [-1, 2, self.grid_size, self.grid_size])
			self.dxy1 = tf.concat([tf.reshape(self.dxy[:, 0, :, :], [-1, self.grid_size, self.grid_size, 1]),
								   tf.reshape(self.dxy[:, 1, :, :], [-1, self.grid_size, self.grid_size, 1])], axis=3)
			# batch,1,y,x
			E_det_j, E_div, E_curl = warp_E(tf.reshape(self.dxy[:, 0, :, :], [-1, self.grid_size, self.grid_size, 1]),
											tf.reshape(self.dxy[:, 1, :, :], [-1, self.grid_size, self.grid_size, 1]))
			source_points = tf.expand_dims(self.source_points, 0)
			theta = source_points + tf.reshape(theta, [-1, 2, self.num_control_points])
			self.theta_vals = tf.reshape(theta, [-1, 2, self.grid_size, self.grid_size])
			x_s, y_s = self._transform(
				inp, theta, self.num_control_points,
				self.pixel_grid, self.num_pixels,
				self.pixel_distances, self.L_inv,
				self.name + '_elastic_transform', batch_size, forward)
			# print('shape:',x_s.shape, y_s.shape)
			# here energy can be calculated from x_s, y_s
			# self.x_s_grid = tf.reshape(x_s,[-1,257,265,1])
			# self.y_s_grid = tf.reshape(y_s,[-1,257,265,1])
			# Energy = warp_E(self.x_s_grid,self.y_s_grid)

			if forward:
				output = _interpolate(
					inp, x_s, y_s,
					self.out_size,
					method=self.interp_method
				)
			else:
				rx_s, ry_s = self._transform(
					inp, theta, self.num_control_points,
					self.pixel_grid, self.num_pixels,
					self.pixel_distances, self.L_inv,
					self.name + '_elastic_transform', forward)
				output = _interpolate(
					inp, rx_s, ry_s,
					self.out_size,
					method=self.interp_method
				)
				pass
		batch_size = tf.shape(inp)[0]
		_, _, _, num_channels = inp.get_shape().as_list()
		output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])
		return output, E_det_j, E_div, E_curl

	def _transform(self, inp, theta, num_control_points, pixel_grid, num_pixels, pixel_distances, L_inv, name,
				   batch_size,
				   forward=True):
		with tf.variable_scope(name):
			# batch_size = inp.get_shape().as_list()[0]

			# Solve as in ref [2]
			theta = tf.reshape(theta, [-1, num_control_points])
			coefficients = tf.matmul(theta, L_inv)
			coefficients = tf.reshape(coefficients, [-1, 2, num_control_points + 3])

			# Transform each point on the target grid (out_size)
			# right_mat = tf.concat(0, [pixel_grid, pixel_distances]) # tensorflow 0.12
			right_mat = tf.concat([pixel_grid, pixel_distances], 0)
			right_mat = tf.tile(tf.expand_dims(right_mat, 0), (tf.shape(inp)[0], 1, 1))  # batch_size
			# transformed_points = tf.batch_matmul(coefficients, right_mat) # tensorflow 0.12
			transformed_points = tf.matmul(coefficients, right_mat)
			transformed_points = tf.reshape(transformed_points, [-1, 2, num_pixels])

			x_s_flat = tf.reshape(transformed_points[:, 0, :], [-1])
			y_s_flat = tf.reshape(transformed_points[:, 1, :], [-1])

			return x_s_flat, y_s_flat

	# U function for the new point and each source point
	@staticmethod
	def U_func(points1, points2):
		# The U function is simply U(r) = r^2 * log(r^2), as in ref [5]_,
		# where r is the euclidean distance
		r_sq = tf.transpose(tf.reduce_sum(tf.square(points1 - points2), axis=0))
		log_r = tf.log(r_sq)
		log_r = tf.where(tf.is_inf(log_r), tf.zeros_like(log_r), log_r)
		phi = r_sq * log_r

		# The U function is simply U(r) = r, where r is the euclidean distance
		# phi = tf.sqrt(r_sq)
		return phi

	@staticmethod
	def get_meshgrid(grid_size_x, grid_size_y):
		# Create 2 x num_points array of source points
		x_points, y_points = tf.meshgrid(
			tf.linspace(-1.0, 1.0, int(grid_size_x)),  # 257
			tf.linspace(-1.0, 1.0, int(grid_size_y)))  # 265
		x_flat = tf.reshape(x_points, (1, -1))
		y_flat = tf.reshape(y_points, (1, -1))
		# points = tf.concat(0, [x_flat, y_flat]) # tensorflow 0.12
		points = tf.concat([x_flat, y_flat], 0)
		return points

	def _initialize_tps(self, source_points, pixel_grid):
		"""
		Initializes the thin plate spline calculation by creating the source
		point array and the inverted L matrix used for calculating the
		transformations as in ref [5]_

		Returns
		----------
		right_mat : float
			Tensor of shape [num_control_points + 3, out_height*out_width].
		L_inv : float
			Tensor of shape [num_control_points + 3, num_control_points].
		source_points : float
			Tensor of shape (2, num_control_points).

		"""

		tL = ElasticTransformer.U_func(tf.expand_dims(source_points, 2), tf.expand_dims(source_points, 1))

		# Initialize L
		# L_top = tf.concat(1, [tf.zeros([2,3]), source_points]) # tensorflow 0.12
		# L_mid = tf.concat(1, [tf.zeros([1, 2]), tf.ones([1, self.num_control_points+1])]) # tensorflow 0.12
		# L_bot = tf.concat(1, [tf.transpose(source_points), tf.ones([self.num_control_points, 1]), tL]) # tensorflow 0.12

		# L = tf.concat(0, [L_top, L_mid, L_bot]) # tensorflow 0.12
		L_top = tf.concat([tf.zeros([2, 3]), source_points], 1)
		L_mid = tf.concat([tf.zeros([1, 2]), tf.ones([1, self.num_control_points + 1])], 1)
		L_bot = tf.concat([tf.transpose(source_points), tf.ones([self.num_control_points, 1]), tL], 1)

		L = tf.concat([L_top, L_mid, L_bot], 0)
		L_inv = tf.matrix_inverse(L)

		# Construct right mat
		to_transform = tf.expand_dims(pixel_grid, 2)
		stacked_source_points = tf.expand_dims(source_points, 1)
		distances = ElasticTransformer.U_func(to_transform, stacked_source_points)

		# Add in the coefficients for the affine translation (1, x, and y,
		# corresponding to a_1, a_x, and a_y)
		ones = tf.ones(shape=[1, self.num_pixels])
		# pixel_distances = tf.concat(0, [ones, distances]) # tensorflow 0.12
		pixel_distances = tf.concat([ones, distances], 0)
		L_inv = tf.transpose(L_inv[:, 3:])

		return pixel_distances, L_inv


def _interpolate(im, x, y, out_size, method):
	if method == 'bilinear':
		return bilinear_interp(im, x, y, out_size)
	# if method == 'bicubic':
	# 	return bicubic_interp(im, x, y, out_size)
	return None


def bilinear_interp(im, x, y, out_size):
	with tf.variable_scope('bilinear_interp'):
		_, height, width, channels = im.get_shape().as_list()
		batch_size = tf.shape(im)[0]
		x = tf.cast(x, tf.float32)
		y = tf.cast(y, tf.float32)
		height_f = tf.cast(height, tf.float32)
		width_f = tf.cast(width, tf.float32)
		out_height = out_size[0]
		out_width = out_size[1]

		# scale indices from [-1, 1] to [0, width/height - 1]
		x = tf.clip_by_value(x, -1, 1)
		y = tf.clip_by_value(y, -1, 1)
		x = (x + 1.0) / 2.0 * (width_f - 1.0)
		y = (y + 1.0) / 2.0 * (height_f - 1.0)

		# do sampling
		x0_f = tf.floor(x)
		y0_f = tf.floor(y)
		x1_f = x0_f + 1
		y1_f = y0_f + 1

		x0 = tf.cast(x0_f, tf.int32)
		y0 = tf.cast(y0_f, tf.int32)
		x1 = tf.cast(tf.minimum(x1_f, width_f - 1), tf.int32)
		y1 = tf.cast(tf.minimum(y1_f, height_f - 1), tf.int32)

		dim2 = width
		dim1 = width * height

		base = _repeat(tf.range(batch_size) * dim1, out_height * out_width)

		base_y0 = base + y0 * dim2
		base_y1 = base + y1 * dim2

		idx_00 = base_y0 + x0
		idx_01 = base_y0 + x1
		idx_10 = base_y1 + x0
		idx_11 = base_y1 + x1

		# use indices to lookup pixels in the flat image and restore
		# channels dim
		im_flat = tf.reshape(im, [-1, channels])

		I00 = tf.gather(im_flat, idx_00)
		I01 = tf.gather(im_flat, idx_01)
		I10 = tf.gather(im_flat, idx_10)
		I11 = tf.gather(im_flat, idx_11)

		# and finally calculate interpolated values
		w00 = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
		w01 = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
		w10 = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
		w11 = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

		output = tf.add_n([w00 * I00, w01 * I01, w10 * I10, w11 * I11])
		return output


def _repeat(x, n_repeats):
	with tf.variable_scope('_repeat'):
		rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
		return tf.reshape(rep, [-1])


# class TrainTForm:
# 	def __init__(self, params):
# 		with tf.device('/gpu:0'):
# 			config = tf.ConfigProto(allow_soft_placement=True)
# 			self.sess = tf.Session(config=config)
#
# 			self.In = tf.placeholder('float', [None, params['width'], params['height'], 2],
# 									 name='In')
# 			self.xytheta = tf.placeholder('float', [None, 3], name='xytheta')
# 			self.moving = tf.slice(self.In, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
# 			self.tformed = self.transformer([params['width'], params['height']])
# 			self.sess.run(tf.global_variables_initializer())
#
# 	def run(self, xytheta, imgs):
# 		feed_dict = {self.In: imgs, self.xytheta: xytheta}
# 		tformed = self.sess.run([self.tformed], feed_dict=feed_dict)
# 		return tformed
#
# 	def transformer(self, out_size, name='SpatialTransformer', **kwargs):
# 		"""Spatial Transformer Layer
#
# 		Implements a spatial transformer layer as described in [1]_.
# 		Based on [2]_ and edited by David Dao for Tensorflow.
#
# 		Parameters
# 		----------
# 		U : float
# 			The output of a convolutional net should have the
# 			shape [num_batch, height, width, num_channels].
# 		theta: float
# 			The output of the
# 			localisation network should be [num_batch, 6].
# 		out_size: tuple of two ints
# 			The size of the output of the network (height, width)
#
# 		References
# 		----------
# 		.. [1]  Spatial Transformer Networks
# 				Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
# 				Submitted on 5 Jun 2015
# 		.. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
#
# 		Notes
# 		-----
# 		To initialize the network to the identity transform init
# 		``theta`` to :
# 			identity = np.array([[1., 0., 0.],
# 								 [0., 1., 0.]])
# 			identity = identity.flatten()
# 			theta = tf.Variable(initial_value=identity)
#
# 		"""
#
# 		def _repeat(x, n_repeats):
# 			with tf.variable_scope('_repeat'):
# 				rep = tf.transpose(
# 					tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
# 				rep = tf.cast(rep, 'int32')
# 				x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
# 				return tf.reshape(x, [-1])
#
# 		def _interpolate(im, x, y, out_size):
# 			with tf.variable_scope('_interpolate'):
# 				# constants
# 				num_batch = tf.shape(im)[0]
# 				height = tf.shape(im)[1]
# 				width = tf.shape(im)[2]
# 				channels = tf.shape(im)[3]
#
# 				x = tf.cast(x, 'float32')
# 				y = tf.cast(y, 'float32')
# 				height_f = tf.cast(height, 'float32')
# 				width_f = tf.cast(width, 'float32')
# 				out_height = out_size[0]
# 				out_width = out_size[1]
# 				zero = tf.zeros([], dtype='int32')
# 				max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
# 				max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
#
# 				# scale indices from [-1, 1] to [0, width/height]
# 				x = (x + 1.0) * (width_f) / 2.0
# 				y = (y + 1.0) * (height_f) / 2.0
#
# 				# do sampling
# 				x0 = tf.cast(tf.floor(x), 'int32')
# 				x1 = x0 + 1
# 				y0 = tf.cast(tf.floor(y), 'int32')
# 				y1 = y0 + 1
#
# 				x0 = tf.clip_by_value(x0, zero, max_x)
# 				x1 = tf.clip_by_value(x1, zero, max_x)
# 				y0 = tf.clip_by_value(y0, zero, max_y)
# 				y1 = tf.clip_by_value(y1, zero, max_y)
# 				dim2 = width
# 				dim1 = width * height
# 				base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
# 				base_y0 = base + y0 * dim2
# 				base_y1 = base + y1 * dim2
# 				idx_a = base_y0 + x0
# 				idx_b = base_y1 + x0
# 				idx_c = base_y0 + x1
# 				idx_d = base_y1 + x1
#
# 				# use indices to lookup pixels in the flat image and restore
# 				# channels dim
# 				im_flat = tf.reshape(im, tf.stack([-1, channels]))
# 				im_flat = tf.cast(im_flat, 'float32')
# 				Ia = tf.gather(im_flat, idx_a)
# 				Ib = tf.gather(im_flat, idx_b)
# 				Ic = tf.gather(im_flat, idx_c)
# 				Id = tf.gather(im_flat, idx_d)
#
# 				# and finally calculate interpolated values
# 				x0_f = tf.cast(x0, 'float32')
# 				x1_f = tf.cast(x1, 'float32')
# 				y0_f = tf.cast(y0, 'float32')
# 				y1_f = tf.cast(y1, 'float32')
# 				wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
# 				wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
# 				wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
# 				wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
# 				output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
# 				return output
#
# 		def _meshgrid(height, width):
# 			with tf.variable_scope('_meshgrid'):
# 				# This should be equivalent to:
# 				#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
# 				#                         np.linspace(-1, 1, height))
# 				#  ones = np.ones(np.prod(x_t.shape))
# 				#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
# 				x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
# 								tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
# 				y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
# 								tf.ones(shape=tf.stack([1, width])))
#
# 				x_t_flat = tf.reshape(x_t, (1, -1))
# 				y_t_flat = tf.reshape(y_t, (1, -1))
#
# 				ones = tf.ones_like(x_t_flat)
# 				grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
# 				return grid
#
# 		def _transform(self, theta, input_dim, out_size):
# 			with tf.variable_scope('_transform'):
# 				num_batch = tf.shape(input_dim)[0]
# 				height = tf.shape(input_dim)[1]
# 				width = tf.shape(input_dim)[2]
# 				num_channels = tf.shape(input_dim)[3]
# 				# theta = tf.reshape(theta, (-1, 2, 3))
# 				# theta = tf.cast(theta, 'float32')
#
# 				DX = tf.slice(theta, [0, 0], [-1, 1]) / (out_size[1] / 2)  # try swap
# 				DY = tf.slice(theta, [0, 1], [-1, 1]) / (out_size[0] / 2)
# 				DRot = (math.pi / 180) * tf.slice(theta, [0, 2], [-1, 1])
#
# 				theta2 = tf.concat([tf.cos(DRot), tf.sin(DRot), DX, -tf.sin(DRot), tf.cos(DRot), DY], axis=1)
# 				theta2 = tf.reshape(theta2, (-1, 2, 3))
# 				self.theta2 = tf.cast(theta2, 'float32')
#
# 				# grid of (x_t, y_t, 1), eq (1) in ref [1]
# 				height_f = tf.cast(height, 'float32')
# 				width_f = tf.cast(width, 'float32')
# 				out_height = out_size[0]
# 				out_width = out_size[1]
# 				grid = _meshgrid(out_height, out_width)
# 				grid = tf.expand_dims(grid, 0)
# 				grid = tf.reshape(grid, [-1])
# 				grid = tf.tile(grid, tf.stack([num_batch]))
# 				self.grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))
#
# 				# Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
# 				T_g = tf.matmul(theta2, self.grid)
# 				x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
# 				y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
# 				self.x_s_flat = tf.reshape(x_s, [-1])
# 				self.y_s_flat = tf.reshape(y_s, [-1])
#
# 				input_transformed = _interpolate(
# 					input_dim, self.x_s_flat, self.y_s_flat,
# 					out_size)
#
# 				output = tf.reshape(
# 					input_transformed, tf.stack([num_batch, out_height, out_width, 1]))
# 				return output
#
# 		with tf.variable_scope(name):
# 			output = _transform(self, self.xytheta, self.moving, out_size)
# 			return output


def ncc(x, y):
	"""
	computes normalized cross correlation
	:param x: input 1
	:param y: input 2
	:return:
	ncc: normalized cross correlation
	"""
	mean_x = tf.reduce_mean(x, [1, 2], keep_dims=True)
	mean_y = tf.reduce_mean(y, [1, 2], keep_dims=True)
	mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3], keep_dims=True)
	mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3], keep_dims=True)
	stddev_x = tf.reduce_sum(tf.sqrt(
		mean_x2 - tf.square(mean_x)), [1, 2, 3], keep_dims=True)
	stddev_y = tf.reduce_sum(tf.sqrt(
		mean_y2 - tf.square(mean_y)), [1, 2, 3], keep_dims=True)
	return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


# def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
# 	"""Batch Spatial Transformer Layer
#
# 	Parameters
# 	----------
#
# 	U : float
# 		tensor of inputs [num_batch,height,width,num_channels]
# 	thetas : float
# 		a set of transformations for each input [num_batch,num_transforms,6]
# 	out_size : int
# 		the size of the output [out_height,out_width]
#
# 	Returns: float
# 		Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
# 	"""
# 	with tf.variable_scope(name):
# 		num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
# 		indices = [[i] * num_transforms for i in xrange(num_batch)]
# 		input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
# 		return transformer(input_repeated, thetas, out_size)


# calculate energy
# difference

"""
helper functions for computing regularization terms, see warp_E
"""
def diff_axis_0(a):
	"""
	1st derivative appproximation for axis 0 of splines
	:param a: a 4D matrix
	:return:
	a 4D matrix of axis 0 derivative approximations
	"""
	return a[:, 1:, :, :] - a[:, :-1, :, :]

def diff_axis_1(a):
	"""
	1st derivative appproximation for axis 1 of splines
	:param a: a 4D matrix
	:return:
	a 4D matrix of axis 1 derivative approximations
	"""
	return a[:, :, 1:, :] - a[:, :, :-1, :]

def conv_axis_0(a):
	filter = tf.constant([1., 1.], shape=[2, 1, 1, 1])
	return tf.nn.conv2d(a, filter, strides=[1, 1, 1, 1], padding='VALID', name='conv_axis_0')

def conv_axis_1(a):
	filter = tf.constant([1., 1.], shape=[1, 2, 1, 1])
	return tf.nn.conv2d(a, filter, strides=[1, 1, 1, 1], padding='VALID', name='conv_axis_1')

def grad_axis_0(a):
	top = diff_axis_0(a[:, :2, :, :])
	body = conv_axis_0(diff_axis_0(a)) / 2
	bottom = diff_axis_0(a[:, -2:, :, :])
	return tf.concat([top, body, bottom], axis=1)

def grad_axis_1(a):
	top = diff_axis_1(a[:, :, :2, :])
	body = conv_axis_1(diff_axis_1(a)) / 2
	bottom = diff_axis_1(a[:, :, -2:, :])
	return tf.concat([top, body, bottom], axis=2)

def warp_E(Sx, Sy):  # ,dims = [8.0,8.0]
	# size = tf.constant([dims[0]*dims[1]],shape = [1])
	gx_x = grad_axis_1(Sx)
	gx_y = grad_axis_0(Sx)
	gy_y = grad_axis_0(Sy)
	gy_x = grad_axis_1(Sy)

	E_divergence = tf.reduce_mean(tf.pow(gx_x + gx_y, 2) + tf.pow(gy_y + gy_x, 2), axis=[1, 2, 3])

	E_curl = tf.reduce_mean(tf.pow(gy_x - gx_y, 2), axis=[1, 2, 3])
	gx_x += 1
	gy_y += 1
	det_j = gx_x * gy_y - gy_x * gx_y
	E_det_j = tf.reduce_mean(tf.pow(det_j - 1, 2), axis=[1, 2, 3])

	return E_det_j, E_divergence, E_curl
