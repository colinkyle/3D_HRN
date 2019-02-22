import imutils
import numpy as np
from imtools import Img
from TF_models import RigidNet, TpsNet
import fsys

## params for neural network
params = {'load_file': None,# full path to model name if starting from pre-trained, ex: '/Users/colin/3D-HRN/model_saves/model-Rigid_265x257'
		  'save_file': 'Rigid',
		  'save_interval': 1000,
		  'batch_size': 32,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.999,  # RMS Prop decay
		  'rms_eps': 1e-9,  # RMS Prop epsilon
		  'width': 265,
		  'height': 257,
		  'numParam': 3,
		  'train': True}

# go to training dir
train_dir = '/Users/colin/3D-HRN/data/30890_/histology/segmented/'
model_dir = '/Users/colin/3D-HRN/model_saves/'
fsys.cd(train_dir)

## initialize net
net = RigidNet(params)

# get file names of training data
train_files = fsys.file('*.png')

## train network

# for moving average cost
avgCost = 100 * [np.inf]
avgCost2 = 100 * [np.inf]

for itrain in range(2000):
	# get batch files
	f_batch = np.random.choice(train_files, params['batch_size'])

	# initialize batch
	batch = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2), dtype=np.float32)

	# load in translated and reflected images
	for b, f in enumerate(f_batch):
		# read image
		img = Img.imread(f)
		# apply random translation
		img = imutils.translate(img, np.random.normal(0,3,1), np.random.normal(0,3,1))

		# apply random flip/reflection
		if np.random.uniform(0, 1, 1) > .5:
			img = np.flipud(img)
		if np.random.uniform(0, 1, 1) > .5:
			img = np.fliplr(img)
		# add images to batch
		batch[b, :, :, :] = np.stack((img, img), axis=2)

	# apply random transformation to batch
	xytheta = np.array([2,2,1])*np.random.uniform(-10,10,(params['batch_size'],3))
	moved, _ = net.transform_with_xytheta(batch,xytheta)

	# restitch batch
	for i in range(params['batch_size']):
		batch[i, :, :, 0] = np.squeeze(moved[i, :, :])

	# apply random noise
	batch = batch + .05 * np.random.randn(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])

	# train
	cnt, squared_error, cost_cc = net.train_supervised(batch,xytheta)

	avgCost.append(squared_error)
	avgCost.pop(0)
	avgCost2.append(cost_cc)
	avgCost2.pop(0)

	print('count: {}, sq_er: {}, cost_cc: {}, avg_sq_er: {}, avg_cost_cc: {}'.format(
		cnt,
		squared_error,
		cost_cc,
		np.mean(avgCost),
		np.mean(avgCost2)))

	if (params['save_file']):
		if cnt % params['save_interval'] == 0:
			net.save_ckpt(model_dir + 'model-' + params['save_file'] + "_" + str(params['width']) + 'x' + str(
				params['height']) + '_' + str(cnt))
			print('Model saved')

