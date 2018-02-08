import sys
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import copy
import cv2
import imutils
import numpy as np
from imtools import Img, reverse_tform_order
import os
from TF_models import AtlasNet, TrainTForm
#from cnn import RegNet, ElasticNet #, KLDiv
import fsys
import math
import matplotlib.pyplot as plt
import scipy.misc
import sys
## TO DO:

## params for neural network
params = {'load_file':'D:/__Atlas__/model_saves/model-regNET_257x265_21000',
          'save_file': 'regNET',
          'save_interval': 10,
          'batch_size': 32,
          'lr': .0001,  # Learning rate
          'rms_decay': 0.9,  # RMS Prop decay
          'rms_eps': 1e-8,  # RMS Prop epsilon
          'width':257,
          'height':265,
          'numParam':3,
          'train':True}

# go to training dir
fsys.cd('D:/__Atlas__')

## initialize net
net = AtlasNet(params)
#net = ElasticNet(params)
#net = ElasticNet(params)

train_files = fsys.file('trainData/*')

# test objective function
# fixed = cv2.imread('30890 B1S2 272N_01_small.jpg')
# fixed = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
# fixed = cv2.resize(fixed, (0,0), fx=0.125, fy=0.125)
# fixed = fixed.astype(np.float32)/255.
# fixed= cv2.copyMakeBorder(fixed,25,25,25,25,cv2.BORDER_CONSTANT,value=0.0)
# moving = fixed
# H1,H2 = net.run(fixed,moving)
# print(H1)
# print(H2)
# print('Done!')
# sys.exit()


# train network
avgCost = 100*[np.inf]
avgCost2 = 100*[np.inf]
train_tform = TrainTForm({'width':257,'height':265})

for itrain in range(10000):
  # get batch files
  f_batch = np.random.choice(train_files,params['batch_size'])

  # random uniform sample of transformations
  ideal = np.random.uniform(low=-40,high=40,size=(params['batch_size'],params['numParam']))
  ideal2 = copy.deepcopy(ideal)
  batch = np.zeros(shape=(params['batch_size'],params['width'],params['height'],2))

  ####
  batch2 = np.zeros(shape=(params['batch_size'], params['width'], params['height'], 2))

  ideal[0,:] = [50,-10,10]
  # create noisy batch
  for b,f in enumerate(f_batch):
    img = cv2.imread(f)
    img[img<5]=255.
    img[img==255.] = 0
    img = cv2.resize(img, (params['height'],params['width']))
    img = img.astype(np.float32) / 255.
    #cyber
    # img = imutils.translate(img, np.random.normal(0,3,1), np.random.normal(0,3,1))


    # cv2.imshow('raw', img)
    # cv2.waitKey(0)

    # do flip/reflection
    # if np.random.uniform(0, 1, 1) > .25:
    #   img = cv2.flip(img,0)
    # if np.random.uniform(0, 1, 1) > .25:
    #   img = cv2.flip(img,1)

    # cv2.imshow('raw', img)
    # cv2.waitKey(0)

    # add noise
    # if np.random.uniform(0, 1, 1) > .25:
    #   m = (0., 0., 0.)
    #   s = (0.1, 0.1, 0.0)
    #   #img = img + cv2.randn(img, m, s)
    #   img = img + .01*np.random.randn(img.shape[0],img.shape[1],img.shape[2])
    # if np.random.uniform(0, 1, 1) > .25:
    #   m = (0., 0., 0.)
    #   s = (0., 0.1, 0.)
    #   #img = img + cv2.randn(img, m, s)
    #   img = img + .01*np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    # if np.random.uniform(0, 1, 1) > .25:
    #   m = (0., 0., 0.)
    #   s = (0.1, 0., 0.)
    #   #img = img + cv2.randn(img, m, s)
    #   img = img + .01*np.random.randn(img.shape[0], img.shape[1], img.shape[2])

    #cv2.imshow('raw', img)
    #cv2.waitKey(0)

    # flip white/ black
    if 0:#np.random.uniform(0, 1, 1) > .25:
      img[img>=.98] = 0.
      img[img <= 0.02] = 1.

    # cv2.imshow('raw', img)
    # cv2.waitKey(0)

    # cv2.imshow('raw', img)
    # cv2.waitKey(0)

    # swap fixed/moving
    # if True:#np.random.uniform(0,1,1)>.5:
    #   fixed = img[:,:,0]
    #   moving = img[:,:,1]
    # else:
    #   fixed = img[:, :, 1]
    #   moving = img[:, :, 0]

    # invert
    if 0:#np.random.uniform(0, 1, 1) > .25:
      fixed = 1. - fixed
    if 0:#np.random.uniform(0, 1, 1) > .25:
      moving = 1. - moving

    # fixed = fixed.astype(np.float32)
    # moving = moving.astype(np.float32)


    # apply transformation
    #DRot = -ideal[b,2]*(math.pi/180)
    #transMAT = np.array([[np.cos(DRot), np.sin(DRot), ideal[b,0]], [-np.sin(DRot), np.cos(DRot), ideal[b,1]]])
    #transMAT = np.linalg.inv(np.array([[np.cos(DRot), np.sin(DRot), -ideal[b,0]], [-np.sin(DRot), np.cos(DRot), -ideal[b,1]], [0, 0, 1]]))
    #transMAT = transMAT[0:2,:]

    #rows, cols = moving.shape
    #moving = cv2.warpAffine(moving, transMAT, (cols, rows))

    # moving = imutils.rotate(moving, ideal[b,2])
    # moving = imutils.translate(moving, ideal[b,0], ideal[b,1])

    #cv2.imshow('raw', fixed)
    #cv2.waitKey(0)
    #cv2.imshow('raw', moving)
    #cv2.waitKey(0)
    # save into batch
    # batch2[b,:,:,:] = cv2.merge((Img.from_array(fixed).p_intensity,Img.from_array(moving).p_intensity))
    #batch[b,:,:,:] = cv2.merge((Img.from_array(img[:,:,0]).p_intensity,Img.from_array(img[:,:,1]).p_intensity))
    batch[b, :, :, :] = cv2.merge((img[:, :, 0], img[:, :, 1]))
    #img[:,:,0:1]#cv2.merge((Img.from_array(fixed).p_intensity,Img.from_array(moving).p_intensity))
    dxdy = reverse_tform_order(ideal[b,2],ideal[b,0],ideal[b,1])
    ideal2[b,0] = dxdy[0][0]
    ideal2[b,1] = dxdy[0][1]
  moved = train_tform.run(-ideal2,batch)[0]
  for i in range(params['batch_size']):
      batch[i,:,:,1] = np.squeeze(moved[i,:,:])

  # flip transform to restore
  #ideal = -ideal
  ## test ElasticNet ##
  # result_ = net.sess.run(net.result,feed_dict={net.In: batch})
  # for i in range(result_.shape[0]):
  #   scipy.misc.imsave('elastic' + str(i) + '.png', result_[i,:,:,0])
  # tformed,xytheta = net.run_batch(batch)
  # for i in range(32):
  #   plt.figure()
  #   merged = np.dstack((np.array(batch[i,:,:,0]),np.array(batch[i,:,:,1]),np.array(batch2[i,:,:,1])))
  #   plt.imshow(merged)
  #   plt.show()
  #
  # plt.imshow(np.array(tformed[0][:][:][:]).squeeze())
  # plt.show()
  cnt, cost, out, cost_t = net.train_param(batch,ideal)
  #train
  #cnt,cost = net.train(batch,ideal)
  avgCost.append(cost)
  avgCost.pop(0)
  avgCost2.append(cost_t)
  avgCost2.pop(0)
  print('count: {}, cost_p: {}, cost_t: {}, avg_cost: {}, avg_cost: {}, sanity: {}'.format(cnt, cost, cost_t, np.mean(avgCost),np.mean(avgCost2),
                                                               np.mean(np.abs(out), axis=0)))
  #print((cnt,cost))

  if (params['save_file']):
    if cnt % params['save_interval'] == 0:
      net.save_ckpt('model_saves/model-' + params['save_file'] + "_" + str(params['width'])+'x'+str(params['height']) + '_' + str(cnt))
      print('Model saved')

# now do test
ideal = np.random.uniform(low=-20, high=20, size=(1, params['numParam']))
#img2 = np.zeros(shape=(1,fixed.shape[0],fixed.shape[1],2))
moving = imutils.rotate(fixed, ideal[0,0])
moving = imutils.translate(moving, ideal[0,1], ideal[0,2])
#img2[0,:,:,:] = cv2.merge((fixed,moving))

#p = net.sess.run(net.Out,feed_dict = {net.In:img2})
result_ = net.sess.run(net.result,feed_dict = {net.In:batch})
#moving = imutils.translate(moving, p[0,1], p[0,2])
#moving = imutils.rotate(moving, p[0,0])
#print(p)
print(-ideal)

cv2.imshow('fixed',batch[0,:,:,0])
cv2.waitKey(0)
cv2.imshow('moving',batch[0,:,:,1])
cv2.waitKey(0) # Waits forever for user to press any key
cv2.imshow('moved',np.swapaxes(np.swapaxes(np.array((batch[0,:,:,0].squeeze(),batch[0,:,:,0].squeeze(),result_[0,:,:,1].squeeze())),0,2),0,1))
cv2.waitKey(0)
cv2.destroyAllWindows()


#  cv2.imshow('fixed',fixed)
# cv2.waitKey(0)
# cv2.imshow('moving',moving)
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()