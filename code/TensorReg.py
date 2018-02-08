import tensorflow as tf
import os
import math
import numpy as np
import cv2
import imutils
from PIL import Image
import numpy as np


# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("../*.jpg"))
#
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)

#my_img = tf.image.convert_image_dtype(tf.image.decode_jpeg(value),dtype=tf.float32) # use png or jpg decoder based on your files.
fixed = cv2.imread('../microscope.tif')
fixed = fixed.astype(np.float32)
# define transformation
#rot_img = tf.contrib.image.rotate(my_img,20*math.pi/180)
moving = imutils.rotate(fixed, 15)
# initialize
init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
with sess.as_default():
    sess.run(init_op)

# Start populating the filename queue.

# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

# for i in range(1): #length of your filename list
#   image = rot_img.eval() #here is your image Tensor :)

#Image.fromarray(np.asarray(image)).show()

# coord.request_stop()
# coord.join(threads)
# fixed = my_img.eval()
# moving2 = rot_img.eval()

sess.close()


sess = tf.Session()
#Define parameters
n_epochs = 1000
learning_rate = 0.01

def my_func(moving,theta):
  return imutils.rotate(moving, theta)

# Define initial tensors
with sess.as_default():
  fixed = tf.constant(fixed,name='fixed')
  #moving = tf.constant(moving,name='moving')
  theta = tf.Variable(initial_value=0.0,dtype=tf.float32,name='theta')
  moved = tf.py_func(my_func, [moving,theta], tf.float32)

#moved = dummyCall(moving,theta)
#error = fixed-moved
#mse = tf.reduce_mean(tf.square(error),name="mse")



#training_op = tf.assign(theta, theta - learning_rate * gradients)

mse = tf.reduce_sum(tf.pow(tf.subtract(fixed, moved), 2))
gradients = tf.gradients(mse, [theta],name='mse')[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)
#rmsprop = tf.train.RMSPropOptimizer(learning_rate).minimize(mse)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
file_writer = tf.summary.FileWriter('.',tf.get_default_graph())

#training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("Epoch", epoch, "MSE =", mse.eval())
    sess.run(training_op)

  best_theta = theta.eval()
