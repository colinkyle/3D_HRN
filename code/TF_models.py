from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import get_config
from tensorflow.python.ops import init_ops
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt

class AtlasNet:
  # def __init__(self, params):
  #   """params has width, depth, numParam"""
  #   self.params = params
  #   self.network_name = 'RegNet'
  #   self.sess = tf.Session()
  #
  #   self.In = tf.placeholder('float', [None, params['width'], params['height'], 2],
  #                           name='In')
  #   self.Ideal = tf.placeholder('float', [None, 3],name='Ideal')
  #
  #   # Layer 1 (Convolutional)
  #   self.conv1 = tf.layers.conv2d(
  #     inputs = self.In,
  #     filters=32,
  #     kernel_size=[5,5],
  #     padding='valid',
  #     kernel_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),
  #     bias_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),
  #     activation=tf.nn.relu)
  #   # Pooling Layer #1
  #   self.pool1 = tf.layers.max_pooling2d(
  #     inputs=self.conv1,
  #     pool_size=[2, 2],
  #     strides=2)
  #
  #   # Convolutional Layer #2
  #   self.conv2 = tf.layers.conv2d(
  #     inputs=self.pool1,
  #     filters=64,
  #     kernel_size=[5, 5],
  #     padding="valid",
  #     kernel_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),
  #     bias_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),
  #     activation=tf.nn.relu)
  #   # Pooling Layer #2
  #   self.pool2 = tf.layers.max_pooling2d(
  #     inputs=self.conv2,
  #     pool_size=[2, 2],
  #     strides=2)
  #
  #   # Convolutional Layer #3
  #   # self.conv3 = tf.layers.conv2d(
  #   #   inputs=self.pool2,
  #   #   filters=128,
  #   #   kernel_size=[5, 5],
  #   #   padding="valid",
  #   #   activation=tf.nn.relu)
  #   # # Pooling Layer #3
  #   # self.pool3 = tf.layers.max_pooling2d(
  #   #   inputs=self.conv3,
  #   #   pool_size=[2, 2],
  #   #   strides=2)
  #   #
  #   # # Convolutional Layer #4
  #   # self.conv4 = tf.layers.conv2d(
  #   #   inputs=self.pool3,
  #   #   filters=64,
  #   #   kernel_size=[5, 5],
  #   #   padding="valid",
  #   #   activation=tf.nn.relu)
  #   # # Pooling Layer #3
  #   # self.pool4 = tf.layers.max_pooling2d(
  #   #   inputs=self.conv4,
  #   #   pool_size=[2, 2],
  #   #   strides=2)
  #
  #   # Dense Layer 1
  #   self.pool4_flat = tf.reshape(self.pool2, [-1, 61*63*64])
  #   self.dense1 = tf.layers.dense(inputs=self.pool4_flat, kernel_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),
  #     bias_initializer=init_ops.VarianceScaling(scale=3.0, mode='fan_out'),units=512, activation=tf.nn.relu)
  #   # self.dropout = tf.layers.dropout(
  #   #   inputs=self.dense1,
  #   #   rate=0.4,
  #   #   training=params['train'])
  #
  #   # Dense Layer 2
  #   self.dense2 = tf.layers.dense(inputs=self.dense1, units=512,activation=tf.nn.relu)
  #   # out is x,y,rot
  #   self.xytheta = tf.layers.dense(inputs=self.dense2, units=3,activation=tf.nn.relu)
  #
  #   self.cost_p = tf.reduce_sum(tf.pow(tf.subtract(self.xytheta, self.Ideal), 2))
  #
  #   # Transformer layer [None, params['width'], params['height'], 2]
  #
  #   self.moving = tf.slice(self.In, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
  #   self.tformed = self.transformer([params['width'], params['height']])
  #
  #   # Cost,Optimizer
  #   #fixed = tf.reshape(tf.slice(self.In, [0, 0, 0, 0], [-1, params['width'], params['height'], 1]),(-1, params['width']*params['height']))
  #   #moved = tf.reshape(self.tformed, (-1, params['width']*params['height']))
  #
  #   fixed = tf.slice(self.In, [0, 0, 0, 0], [-1, params['width'], params['height'], 1]),(-1, params['width']*params['height'])
  #   #self.cost_t = tf.reduce_sum(tf.pow(tf.subtract(fixed, moved), 2))
  #   #self.cost_i = tf.reduce_sum(tf.multiply(fixed, moved),axis=1) / tf.norm(fixed,axis=1) / tf.norm(moved,axis=1)
  #   #self.cost_t = -tf.reduce_sum(self.cost_i)
  #   self.cost_t = ncc(fixed[0], self.tformed)
  #
  #
  #   if self.params['load_file'] is not None:
  #     self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step', trainable=False)
  #   else:
  #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
  #
  #   # Gradient descent on loss function
  #   self.optim = tf.train.AdamOptimizer(params['lr'])
  #   self.cc_train = self.optim.minimize(-self.cost_t,global_step=self.global_step)
  #
  #   #self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(self.cost_t,
  #   #                                                                                                     global_step=self.global_step)
  #
  #   self.saver = tf.train.Saver(max_to_keep=0)
  #
  #   self.sess.run(tf.global_variables_initializer())
  #
  #   if self.params['load_file'] is not None:
  #     print('Loading checkpoint...')
  #     self.saver.restore(self.sess, self.params['load_file'])
  def __init__(self, params):
    with tf.device('/gpu:0'):
        """params has width, depth, numParam"""
        self.params = params
        self.network_name = 'AtlasNet'
        ## GPU Session stuff
        ## init main
        # flags = tf.app.flags
        #
        # # Model
        # flags.DEFINE_string('model', 'm1', 'Type of model')
        #
        # # Environment
        # flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')
        #
        # # Etc
        # flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
        # flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
        # flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
        # flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
        # flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
        #
        # FLAGS = flags.FLAGS
        #
        # # Set random seed
        # tf.set_random_seed(FLAGS.random_seed)
        # random.seed(FLAGS.random_seed)

        # gpu_options = tf.GPUOptions(
        #     per_process_gpu_memory_fraction=0.8)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess.config = get_config(FLAGS) or FLAGS
        # if not tf.test.is_gpu_available() and FLAGS.use_gpu:
        #     raise Exception("use_gpu flag is true when no GPUs are available")
        # if not FLAGS.use_gpu:
        #     self.sess.config.cnn_format = 'NHWC'
        ## end gpu session stuff
        #self.sess = tf.Session()
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # Runs the op.
        #print(self.sess.run(c))

        self.In = tf.placeholder('float', [None, params['width'], params['height'], 2],
                                name='In')
        self.Ideal = tf.placeholder('float', [None, params['numParam']], name=self.network_name + '_Ideal')

        # Layer 1 (Convolutional)
        layer_name = 'conv1'
        size = 5
        channels = 2
        filters = 32
        stride = 5
        self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                              name=self.network_name + '_' + layer_name + '_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c1 = tf.nn.conv2d(self.In, self.w1, strides=[1, stride, stride, 1], padding='SAME',
                               name=self.network_name + '_' + layer_name + '_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name + '_' + layer_name + '_activations')

        # Layer 2 (Convolutional)
        layer_name = 'conv2'
        size = 5
        channels = 32
        filters = 64
        stride = 5
        self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                              name=self.network_name + '_' + layer_name + '_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',
                               name=self.network_name + '_' + layer_name + '_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), name=self.network_name + '_' + layer_name + '_activations')

        o2_shape = self.o2.get_shape().as_list()

        # Layer 3 (Fully connected)
        layer_name = 'fc3'
        hiddens = 256
        dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
        self.w3 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                              name=self.network_name + '_' + layer_name + '_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat, self.w3), self.b3, name=self.network_name + '_' + layer_name + '_ips')
        self.o3 = tf.nn.relu(self.ip3, name=self.network_name + '_' + layer_name + '_activations')

        # Layer 4 (Fully connected, parameter output)
        layer_name = 'fc4'
        hiddens = params['numParam']
        dim = 256
        self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                              name=self.network_name + '_' + layer_name + '_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
        self.xytheta = tf.add(tf.matmul(self.o3, self.w4), self.b4, name='Out')

        #### Cost,Optimizer train on parameters (supervised) ####
        self.cost_sup = tf.reduce_sum(tf.pow(tf.subtract(self.xytheta, self.Ideal), 2))

        if self.params['load_file'] is not None:
          self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step', trainable=False)
        else:
          self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Gradient descent on loss function
        self.rmsprop_sup = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(self.cost_sup,
                                                                                                             global_step=self.global_step)
        #### Do registration ####
        self.moving = tf.slice(self.In, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
        self.tformed = self.transformer([params['width'], params['height']])

        fixed = tf.slice(self.In, [0, 0, 0, 0], [-1, params['width'], params['height'], 1]),(-1, params['width']*params['height'])

        #### Cost,Optimizer train on similarity function (unsupervised) ####
        self.cost_cc = ncc(fixed[0], self.tformed)
        self.cost_z = tf.reduce_sum(tf.pow(tf.subtract(self.tformed, fixed[0]), 2), name='RMS')
        self.rmsprop_cc = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(self.cost_z,
                                                                                                                global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
          print('Loading checkpoint...')
          self.saver.restore(self.sess, self.params['load_file'])

        #self.sess.run(tf.variables_initializer([self.rmsprop_cc]))
        momentum_initializers = [var.initializer for var in tf.global_variables() if 'RMS' in var.name]
        self.sess.run(momentum_initializers)

  def train(self, bat_In, param):
    feed_dict = {self.In: bat_In, self.Ideal:param}
    _, cnt, cost, out, cost_t = self.sess.run([self.rmsprop_cc, self.global_step, self.cost_sup, self.xytheta, self.cost_cc], feed_dict=feed_dict)
    return cnt, cost/bat_In.shape[0], out, cost_t

  def train_param(self, bat_In, param):
    feed_dict = {self.In: bat_In, self.Ideal:param}
    _, cnt, cost, out, cost_t = self.sess.run([self.rmsprop_sup, self.global_step, self.cost_sup, self.xytheta, self.cost_cc], feed_dict=feed_dict)
    #print(self.sess.run([self.rmsprop_sup, self.global_step, self.cost_sup, self.xytheta, self.cost_cc], feed_dict=feed_dict))
    return cnt, cost / bat_In.shape[0], out, cost_t

  def save_ckpt(self, filename):
    self.saver.save(self.sess, filename)

  def run_batch(self,batch):
    import matplotlib.pyplot as plt
    import numpy as np
    feed_dict = {self.In:batch}
    tformed,xytheta = self.sess.run([self.tformed,self.xytheta], feed_dict=feed_dict)

    # plt.imshow(np.array(batch[0,:,:,1]).squeeze())
    # plt.show()
    #
    # plt.imshow(np.array(tformed[0][:][:][:]).squeeze())
    # plt.show()
    return tformed,xytheta
    #print(xytheta)
    #print('got tform')

  def transformer(self,out_size, name='SpatialTransformer', **kwargs):
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
              x = (x + 1.0)*(width_f) / 2.0
              y = (y + 1.0)*(height_f) / 2.0

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
              dim1 = width*height
              base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
              base_y0 = base + y0*dim2
              base_y1 = base + y1*dim2
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
              wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
              wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
              wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
              wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
              output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
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

      def _transform(self,theta, input_dim, out_size):
          with tf.variable_scope('_transform'):
              num_batch = tf.shape(input_dim)[0]
              height = tf.shape(input_dim)[1]
              width = tf.shape(input_dim)[2]
              num_channels = tf.shape(input_dim)[3]
              #theta = tf.reshape(theta, (-1, 2, 3))
              #theta = tf.cast(theta, 'float32')

              DX = tf.slice(theta,[0, 0],[-1, 1])/(out_size[1]/2)#try swap
              DY = tf.slice(theta,[0, 1],[-1, 1])/(out_size[0]/2)
              DRot = (math.pi/180)*tf.slice(theta,[0, 2],[-1, 1])

              theta2 = tf.concat([tf.cos(DRot), tf.sin(DRot), DX, -tf.sin(DRot), tf.cos(DRot), DY], axis=1)
              theta2 = tf.reshape(theta2, (-1, 2, 3))
              self.theta2 = tf.cast(theta2, 'float32')

              # grid of (x_t, y_t, 1), eq (1) in ref [1]
              height_f = tf.cast(height, 'float32')
              width_f = tf.cast(width, 'float32')
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
          output = _transform(self,self.xytheta, self.moving, out_size)
          return output

class TrainTForm:
    def __init__(self,params):
        with tf.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)

            self.In = tf.placeholder('float', [None, params['width'], params['height'], 2],
                                name='In')
            self.xytheta = tf.placeholder('float', [None, 3],name='xytheta')
            self.moving = tf.slice(self.In, [0, 0, 0, 1], [-1, params['width'], params['height'], 1])
            self.tformed = self.transformer([params['width'], params['height']])
            self.sess.run(tf.global_variables_initializer())

    def run(self,xytheta,imgs):
        feed_dict = {self.In: imgs,self.xytheta:xytheta}
        tformed = self.sess.run([self.tformed], feed_dict=feed_dict)
        return tformed

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
                num_channels = tf.shape(input_dim)[3]
                # theta = tf.reshape(theta, (-1, 2, 3))
                # theta = tf.cast(theta, 'float32')

                DX = tf.slice(theta, [0, 0], [-1, 1]) / (out_size[1] / 2)  # try swap
                DY = tf.slice(theta, [0, 1], [-1, 1]) / (out_size[0] / 2)
                DRot = (math.pi / 180) * tf.slice(theta, [0, 2], [-1, 1])

                theta2 = tf.concat([tf.cos(DRot), tf.sin(DRot), DX, -tf.sin(DRot), tf.cos(DRot), DY], axis=1)
                theta2 = tf.reshape(theta2, (-1, 2, 3))
                self.theta2 = tf.cast(theta2, 'float32')

                # grid of (x_t, y_t, 1), eq (1) in ref [1]
                height_f = tf.cast(height, 'float32')
                width_f = tf.cast(width, 'float32')
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

def ncc(x, y):
  mean_x = tf.reduce_mean(x, [1,2], keep_dims=True)
  mean_y = tf.reduce_mean(y, [1,2], keep_dims=True)
  mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
  mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
  stddev_x = tf.reduce_sum(tf.sqrt(
    mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
  stddev_y = tf.reduce_sum(tf.sqrt(
    mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
  return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer

    Parameters
    ----------

    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]

    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)