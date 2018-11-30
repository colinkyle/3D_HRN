import fsys
import tensorflow as tf
from TF_models import AtlasNet, ElasticNet
import matplotlib.pyplot as plt
import numpy as np

fsys.cd('D:/__Atlas__')
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
netR = AtlasNet(params)
params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic2_257x265_146000'
params['save_file'] = 'Elastic2'
netE = ElasticNet(params)

weights = netE.sess.run(netE.w1)

if 1:
	title = ['fixed', 'moving']
	fig, axes = plt.subplots(weights.shape[3], 2, figsize=(3, 30))
	for ii in range(weights.shape[3]):

		for idx in range(weights.shape[2]):
			i = 0  # idx % 4  # Get subplot row
			j = idx  # // 4
			img = weights[:, :, idx, ii].squeeze()
			img = img + np.min(img)
			img = img / np.max(img)

			# img = np.concatenate([img,np.zeros((15,15,1))],axis=2)

			axes[ii, j].imshow(img)
			axes[ii, j].set_title(title[j])
			axes[ii, j].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
									left='off', labelleft='off')
	# plt.axis('off')
	# plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("l1_elastic.pdf", format='pdf')
	plt.show()

weights = netE.sess.run(netE.w2)
if 1:
	fig, axes = plt.subplots(weights.shape[2], 24, figsize=(30, 30))
	for ii in range(weights.shape[2]):

		for idx in range(24):  # range(weights.shape[3]):
			i = 0  # idx % 4  # Get subplot row
			j = idx  # // 4
			img = weights[:, :, ii, idx].squeeze()
			img = img + np.min(img)
			img = img / np.max(img)

			# img = np.concatenate([img,np.zeros((15,15,1))],axis=2)

			axes[ii, j].imshow(img)
			# axes[ii, j].set_title(title[j])
			axes[ii, j].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off',
									left='off', labelleft='off')
	# plt.axis('off')
	# plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("l2_elastic.pdf", format='pdf')
	plt.show()
