import fsys
from imtools import Img, ecc
import numpy as np
from TF_models import AtlasNet
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interpolate
from scipy import stats


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
	return ul, sl


def fit_hypsecant(x, y, pow):
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
	return ul, sl


mpl.rcParams['pdf.fonttype'] = 42

fsys.cd('D:/__Atlas__/data/35717/histology/segmented')

files = fsys.file('*.png')

fixed = Img.imread(files[25])
moving = Img.imread(files[26])

params = {'load_file': 'D:/__Atlas__/model_saves/model-regNETshallow_265x257_512000',
		  'save_file': 'regNETshallow',
		  'save_interval': 1000,
		  'batch_size': 80,
		  'lr': .0001,  # Learning rate
		  'rms_decay': 0.9,  # RMS Prop decay
		  'rms_eps': 1e-8,  # RMS Prop epsilon
		  'width': 265,
		  'height': 257,
		  'numParam': 3,
		  'train': True}

netR = AtlasNet(params)
batch = np.zeros((1, 265, 257, 2))
batch[0] = np.stack((fixed, moving), axis=2)

tformed, xytheta, cost_cc = netR.run_batch(batch)
batch[0, :, :, 1] = tformed.squeeze()

plt.imshow(np.concatenate((batch.squeeze(), np.zeros((265, 257, 1))), axis=2))
plt.show()

Xform = np.zeros((64000, 3))
cnt = 0
r = [30, 30, 10]
num = 41
for i in np.linspace(-r[0], r[0], 40):
	for j in np.linspace(-r[1], r[1], 40):
		for k in np.linspace(-r[2], r[2], 40):
			Xform[cnt] = [i, j, k]
			cnt += 1

batch = np.zeros((80, 265, 257, 2))
for i in range(80):
	batch[i] = np.stack((fixed, tformed[0].squeeze()), axis=2)

CC = np.zeros((40, 40, 40))
y, x, z = np.meshgrid(range(40), range(40), range(40))
y = y.flatten()
x = x.flatten()
z = z.flatten()

for b in range(800):
	print(Xform[b * 80:b * 80 + 80])
	tformed, xytheta, theta2, cost_cc = netR.transform_batch(batch, Xform[b * 80:b * 80 + 80])
	X, Y, Z = x[b * 80:b * 80 + 80], y[b * 80:b * 80 + 80], z[b * 80:b * 80 + 80]
	for i in range(tformed.shape[0]):
		CC[X[i], Y[i], Z[i]] = ecc(fixed, tformed[i])

dist_x = np.max(np.max(CC, axis=2, keepdims=True), axis=1, keepdims=True).flatten()
dist_y = np.max(np.max(CC, axis=2, keepdims=True), axis=0, keepdims=True).flatten()
dist_theta = np.max(np.max(CC, axis=1, keepdims=True), axis=0, keepdims=True).flatten()
dist_x = np.concatenate((dist_x, [ecc(fixed, batch[0, :, :, 1])]), axis=0)
dist_y = np.concatenate((dist_y, [ecc(fixed, batch[0, :, :, 1])]), axis=0)
dist_theta = np.concatenate((dist_theta, [ecc(fixed, batch[0, :, :, 1])]), axis=0)

x = np.linspace(-r[0], r[0], 40)
y = np.linspace(-r[1], r[1], 40)
z = np.linspace(-r[2], r[2], 40)
x = np.concatenate([x, [0]], axis=0)
y = np.concatenate([y, [0]], axis=0)
z = np.concatenate([z, [0]], axis=0)
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

fig, ax = plt.subplots(3, 1, figsize=(5, 15))
ax[0].plot(x, dist_x)

u_x, s_x = fit_laplace(x, dist_x)
dist_x_hat = stats.laplace.pdf(x, loc=u_x, scale=s_x)
dist_x_hat = np.max(dist_x) * dist_x_hat / np.max(dist_x_hat)
ax[0].plot(x, dist_x_hat, c=[0, 1, 0])
ax[0].set_xlabel('$\Delta$x (pixels)')
ax[0].set_ylabel('max similarity')
ax[1].plot(y, dist_y)

u_y, s_y = fit_laplace(y, dist_y)
dist_y_hat = stats.laplace.pdf(y, loc=u_y, scale=s_y)
dist_y_hat = np.max(dist_y) * dist_y_hat / np.max(dist_y_hat)

u_y, s_y = fit_hypsecant(y, dist_y, .1)
dist_y_hat2 = np.power(stats.hypsecant.pdf(y, loc=u_y, scale=s_y), .1)
dist_y_hat2 = np.max(dist_y) * dist_y_hat2 / np.max(dist_y_hat2)

ax[1].plot(y, dist_y_hat, c=[0, 1, 0])
ax[1].plot(y, dist_y_hat2, c=[1, 0, 0])
ax[1].set_xlabel('$\Delta$y (pixels)')
ax[1].set_ylabel('max similarity')

ax[2].plot(z, dist_theta)

u_z, s_z = fit_laplace(z, dist_theta)
dist_z_hat = stats.laplace.pdf(z, loc=u_z, scale=s_z)
dist_z_hat = np.max(dist_theta) * dist_z_hat / np.max(dist_z_hat)

u_z, s_z = fit_hypsecant(z, dist_theta, .01)
dist_z_hat2 = np.power(stats.hypsecant.pdf(z, loc=u_z, scale=s_z), .01)
dist_z_hat2 = np.max(dist_theta) * dist_z_hat2 / np.max(dist_z_hat2)
print(s_z)
ax[2].plot(z, dist_z_hat, c=[0, 1, 0])
ax[2].plot(z, dist_z_hat2, c=[1, 0, 0])
ax[2].set_xlabel(r'$\Delta\theta$  (degrees)')
ax[2].set_ylabel('max similarity')

plt.savefig("Rigid_Distributions.pdf", format='pdf')
plt.show()
