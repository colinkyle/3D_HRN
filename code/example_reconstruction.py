import copy
import matplotlib.pyplot as plt
import numpy as np
import re
import nibabel as nib

import fsys
from imtools import Stack, Template, get_defaultMETA, load_obj, save_obj
from search import iterativePowellWithRetrain, iterativePowellWithRetrainRigid
from TF_models import RigidNet, TpsNet

# SET PATHS, PARAMETERS, AND WHICH PRE-TRAINED NETWORKS TO LOAD
root_dir = '/Users/Colin/3D-HRN/'
fsys.cd(root_dir+'data/30890_/histology/segmented')

rigid_params = {'load_file': root_dir+'model_saves/model-Rigid_265x257',
			  'save_file': 'Rigid',
			  'save_interval': 1000,
			  'batch_size': 32,
			  'lr': .0001,  # Learning rate
			  'rms_decay': 0.9,  # RMS Prop decay
			  'rms_eps': 1e-8,  # RMS Prop epsilon
			  'width': 265,
			  'height': 257,
			  'numParam': 3,
			  'train': True}
elastic_params = copy.deepcopy(rigid_params)
elastic_params['load_file'] = root_dir+'model_saves/model-TPS_265x257'
elastic_params['save_file'] = 'TPS'

# GENERATE STACK WITH EDGE DISTRIBUTIONS
if True:
	print('Creating image stack...')
	# histology section files
	fList = fsys.file('*.png')

	# Generate metadata list with each section's metadata.
	# Each section has a file name and 3D affine matrix:
	# for resolution and z_location.  STN code does not
	# consider each image's affine matrix.  Rigid STN
	# rotates about image center.  affine matrixes are
	# used to resample the Template to matching resolution
	# at the precise z_location to match each section.
	metaList = []
	pixel_size = 0.244
	section_size = .12 # z spacing
	#file names contain section locations (this parses)
	locs = [int(re.search('[0-9]*[N]', fList[i]).group(0)[0:-1]) for i in range(len(fList)) if
			re.search('[0-9]*[N]', fList[i])]
	z_middle = (max(locs) + min(locs)) / 2.
	for i, f in enumerate(fList):
		meta = get_defaultMETA()
		meta['fname'] = f
		# set x
		meta['MAT'][0][0] = pixel_size
		meta['MAT0'][0][0] = pixel_size
		# set y
		meta['MAT'][1][1] = pixel_size
		meta['MAT0'][1][1] = pixel_size
		# set z
		meta['MAT'][2][2] = section_size
		meta['MAT0'][2][2] = section_size

		# set x offset
		meta['MAT'][0][3] = -pixel_size * 265 / 2
		meta['MAT0'][0][3] = -pixel_size * 257 / 2

		# set y offset
		meta['MAT'][1][3] = -pixel_size * 265 / 2
		meta['MAT0'][1][3] = -pixel_size * 257 / 2

		# set z offset
		z_pos = int(re.search('[0-9]*[N]', f).group(0)[0:-1])
		z_mm = section_size * (z_pos - z_middle)
		meta['MAT'][2][3] = z_mm
		meta['MAT0'][2][3] = z_mm

		metaList.append(meta)

	stack = Stack(fList, metaList)
	print('Estimating edge distributions...')
	stack.estimate_edge_distributions(rigid_params)
	save_obj(stack, '30890_hist.obj')
	print('Done!')

# load template for reconstruction and viewing
print('Loading template and stack...')
fsys.cd(root_dir+'data/30890_/mri')
template = Template('_30890_.nii.gz')

fsys.cd(root_dir+'data/30890_/histology/segmented')
stack = load_obj('30890_hist.obj')

# POWELL SEARCH for RECONSTRUCTION PARAMS Psi_T
if True:
	#Do full search
	netR = RigidNet(rigid_params)
	netE = TpsNet(elastic_params)

	print('starting full search...')
	Psi_T, netR, netE = iterativePowellWithRetrain(stack,template,netR,netE,[5,-5,-5,0.95],niter = 10)
	print('finished search...')

	# Alternatively...
	# Do rigid search
	# netR = RigidNet(rigid_params)
	#
	# print('starting rigid search...')
	# Psi_T, netR = iterativePowellWithRetrainRigid(stack, template, netR, [5, -5, -5, 0.95], niter=10)
	# print('finished search...')


# SAVE AND VIEW RESULTS
if True:
	# check if Psi_T was generated in search above
	if 'Psi_T' in locals():
		z, theta_x, theta_y, dxyz = Psi_T
	else:# choose Psi_T locations manually
		z = 1.65
		theta_x = -6
		theta_y = -0.625
		dxyz = 0.9125

	#save resampled template
	new_mri = template.get_nifti3D_from_warped_template(stack, z, theta_x, theta_y, dxyz, dxyz)
	fsys.cd(root_dir + 'data/30890_/mri')
	nib.save(new_mri, 'template_resampled.nii.gz')

	#retrain rigid network to optimize reigstration to final template position
	# (Save the model after training for faster viewing next time)
	netR = RigidNet(rigid_params)
	batch = template.gen_batch(stack, z, theta_x, theta_y, dxyz, dxyz)
	netR = template.retrain_TF(netR, batch, ntrain=100, nbatch=32)

	#generate rigidly reconstructed stack
	tformed, xytheta, cost_cc = netR.run(batch)
	for i in range(batch.shape[0]):
		batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])

	# retrain non-rigid elastic network to optimize reigstration to final template position
	netE = TpsNet(elastic_params)
	netE = template.retrain_TF_E(netE, batch, ntrain=100, nbatch=32)
	moved, theta_mri, cost_cc, full_cost, def_energy_cost = netE.run(batch)

	# save 3D nifti of reconstructed histology
	stack.set_warped_from_moved(moved)
	hist_mri = stack.get_nifti3D_from_warped_stack()
	fsys.cd(root_dir + 'data/30890_/mri')
	nib.save(hist_mri, 'hist_reconstructed.nii.gz')

	# plot each histology template pair
	for i in range(batch.shape[0]):
		plt.figure(figsize=(15, 15))
		plt.subplot(2, 2, 1)
		# rescale in case range out of [0,1]
		batch[i, :, :, 0] = batch[i, :, :, 0] - np.min(batch[i, :, :, 0])
		batch[i, :, :, 1] = batch[i, :, :, 1] - np.min(batch[i, :, :, 1])
		moved[i, :, :, 0] = moved[i, :, :, 0] - np.min(moved[i, :, :, 0])

		batch[i, :, :, 0] = batch[i, :, :, 0] / np.max(batch[i, :, :, 0])
		batch[i, :, :, 1] = batch[i, :, :, 1] / np.max(batch[i, :, :, 1])
		moved[i,:,:,0] = moved[i,:,:,0] / np.max(moved[i,:,:,0])

		merged = np.dstack(
			(np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 0])))
		plt.imshow(merged)

		plt.subplot(2, 2, 2)
		merged = np.dstack(
			(np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1])))
		plt.imshow(merged)

		plt.subplot(2, 2, 3)
		merged = np.dstack(
			(np.array(batch[i, :, :, 0]), np.array(moved[i, :, :, 0]), np.array(moved[i, :, :, 0])))
		plt.imshow(merged)

		plt.show()
