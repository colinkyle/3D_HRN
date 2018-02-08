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
#import pymp
from scipy.interpolate import interpn
from scipy import stats
from TF_models import AtlasNet
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import tifffile
import fsys

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
  def imread(cls, fpath,metadata=None):
    # read META data
    with tifffile.TiffFile(fpath) as tif:
      data = tif.asarray()
      data[data==255.0] = 0
      data = data/255.0
      data = rgb2gray(data)
      # image data from file overrides passed metadata
      if hasattr(tif[0],'image_description'):
        metadata = tif[0].image_description
        metadata = json.loads(metadata.decode('utf-8'))
      elif metadata:
        pass
      else:
        metadata = get_defaultMETA()
    return cls.from_array(data, metadata)

  @classmethod
  def from_array(cls, imgdat, metadata={}):
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

  def compose_translation_3D(self,xyz):
    affine = np.array([1,0,0,xyz[0]],[0,1,0,xyz[1]],[0,0,1,xyz[2]],[0,0,0,1])
    return np.dot(np.array(self.meta['MAT']), affine)

  def compose_rotation_3D(self,theta_xyz):
    affine_x = np.array([[1, 0, 0, 0], ...
                        [0, math.cos(math.radians(theta_xyz[0])), -1.*math.sin(math.radians(theta_xyz[0])), 0], ...
                        [0, math.sin(math.radians(theta_xyz[0])), math.cos(math.radians(theta_xyz[0])), 0], ...
                        [0, 0, 0, 1]])

    affine_y = np.array([[math.cos(math.radians(theta_xyz[1])), 0, math.sin(math.radians(theta_xyz[1])), 0], ...
                        [0, 1, 0, 0], ...
                        [-1.*math.sin(math.radians(theta_xyz[1])), 0, math.cos(math.radians(theta_xyz[1])), 0], ...
                        [0, 0, 0, 1]])

    affine_z = np.array([[math.cos(math.radians(theta_xyz[2])), -1.*math.sin(math.radians(theta_xyz[2])), 0, 0], ...
                        [math.sin(math.radians(theta_xyz[2])), math.cos(math.radians(theta_xyz[2])), 0, 0], ...
                        [0, 0, 1, 0], ...
                        [0, 0, 0, 1]])

    affine = np.dot(np.dot(affine_x,affine_y),affine_z)
    return np.dot(np.array(self.meta['MAT']), affine)

  def compose_rigid_2D(self,xytheta):
    #affine_og = self.get_affine_2D()
    affine = np.array([[math.cos(math.radians(xytheta[2])), math.sin(math.radians(xytheta[2])), xytheta[0]],
                        [-1.0*math.sin(math.radians(xytheta[2])), math.cos(math.radians(xytheta[2])), xytheta[1]],
                        [0, 0, 1]])

    return affine #np.dot(affine_og, np.dot(affine_rot, affine_trans))

  def write_affine(self,MAT):
    self.meta['MAT'] = MAT

  def get_affine_2D(self):
    affine = self.meta['MAT']
    row1 = [affine[0][i] for i in [0,1,3]]
    row2 = [affine[1][i] for i in [0,1,3]]
    row3 = [0, 0, 1]
    return np.array([row1, row2, row3])

  @lazy_property
  def p_intensity(self):
    # Get all relatives
    binCount,bins = np.histogram(self,bins=256,range=[machineEpsilon(np.float32),1.])
    binCount = binCount/max(binCount)
    p_intensity = np.zeros(shape=self.shape[:2],dtype=np.float32)
    l = bins[0]
    ibin = 0
    for h in bins[1:]:
      p_intensity[np.logical_and(self>=l,self<h)] = binCount[ibin]
      ibin+=1
      l = h
    p_intensity[self==h] = binCount[-1]
    #p_intensity = p_intensity/np.max(p_intensity)
    return p_intensity

# defines relationship between images in terms of
# probability distributions of relative positions
class Edge:
  def __init__(self,img0,img1):
    if (not isinstance(img0, Img)) or (not isinstance(img1, Img)):
      raise TypeError("Edge must reference two Img objects")
    self.node0 = img0
    self.node1 = img1

    self.dz = img1.meta['MAT0'][2][3] - img0.meta['MAT0'][2][3]

    self.param_x = [-self.node1.shape[0]/2, 0, self.node1.shape[0]/2]
    self.param_y = [-self.node1.shape[0]/2, 0, self.node1.shape[0]/2]
    self.param_theta = [0, 6]

  def p_x(self,dx):

    if self.param_x[0]:
      c = (self.param_x[1] - self.param_x[0]) / (self.param_x[2] - self.param_x[0])
      loc = self.param_x[0]
      scale = self.param_x[2] - self.param_x[0]
      return stats.triang.pdf(dx, c=c, loc=loc, scale=scale)/(stats.triang.pdf(self.param_x[1], c=c, loc=loc, scale=scale) + machineEpsilon() )
    else:
      return None

  def p_y(self,dy):

    if self.param_y[0]:
      c = (self.param_y[1] - self.param_y[0]) / (self.param_y[2] - self.param_y[0])
      loc = self.param_y[0]
      scale = self.param_y[2] - self.param_y[0]
      return stats.triang.pdf(dy, c=c, loc=loc, scale=scale)/(stats.triang.pdf(self.param_y[1], c=c, loc=loc, scale=scale) + machineEpsilon() )
    else:
      return None

  def p_theta(self,dtheta):
    if self.param_theta[0]:
      return stats.laplace.pdf(dtheta, loc=self.param_theta[0], scale=self.param_theta[1])/stats.laplace.pdf(self.param_theta[0], loc=self.param_theta[0], scale=self.param_theta[1])
    else:
      return None

  def sample_xytheta(self,nsamples = 500):
    x = np.random.triangular(self.param_x[0],self.param_x[1],self.param_x[2],size=nsamples)

    y = np.random.triangular(self.param_y[0],self.param_y[1],self.param_y[2],size=nsamples)

    theta = np.random.laplace(self.param_theta[0],self.param_theta[1],size=nsamples)

    return [x,y,theta]

  def view(self,img0=None, img1=None, img2=None):
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

    big_size = [max([sz0[0],sz1[0],sz2[0]]), max([sz0[1],sz1[1],sz2[1]]), 3]

    new_img = np.zeros(big_size)
    new_img[0:sz0[0],0:sz0[1],0] = img0
    new_img[0:sz1[0],0:sz1[1], 1] = img1
    new_img[0:sz2[0], 0:sz2[1], 2] = img2

    plt.imshow(np.ndarray(
        shape=big_size,
        buffer=new_img.astype(np.float32),
        dtype=np.float32), cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

# collection class
class Img3D:
  def __init__(self, fList = [], metaList = []):
    """Img3D is container of Img objects, fnames, and edge dictionary"""
    self.fList = fList
    self.images = []
    self.edges = {}

    if self.fList:
      for i,fname in enumerate(self.fList):
        self.images.append(Img.imread(fname))
        self.images[-1].meta['fname'] = fname
    if metaList:
      self.update_meta(metaList)

  def save(self,fname):
    save_obj(self,fname)

  def update_meta(self,metaList):
    if len(metaList) == len(self.images):
      for i,meta in enumerate(metaList):
        self.images[i].meta.update(meta)
    else:
      raise ValueError('metaList must be same length as self.images')

  def build_edge_net(self,n_neighbors):
    for i in range(len(self.images)-n_neighbors):
      for j in np.arange(i+1,i+n_neighbors+1):
        self.edges[(i,j)] = Edge(self.images[i],self.images[j])

  def estimate_edge_distributions(self,nsamples = 1000):
    # Loop edges
    for edge_id,edge in self.edges.items():
      print(edge_id)
      # Search for best registration
      (cc, warp) = self.rigid_reg(edge_id)

      dy,dx,dtheta = warp[0][2],warp[1][2],np.arcsin(warp[0][1])*180/np.pi
      print('max: ', dx,dy,dtheta,'cc: ',cc)

      # Generate to model full distribution around rigid warp parameters
      X,Y,Theta = edge.sample_xytheta(nsamples=nsamples)
      p_xytheta = np.zeros((nsamples,))
      #with pymp.Parallel(4) as p:
      for index in range(nsamples):
        transMAT = edge.node1.compose_rigid_2D([Y[index], X[index], Theta[index]])
        rows, cols = edge.node1.shape
        node1_transformed = cv2.warpAffine(edge.node1.p_intensity, transMAT[0:2, :], (cols, rows))
        p_xytheta[index] = ecc(node1_transformed, self.edges[edge_id].node0.p_intensity)

      # clean up in case nan
      X = X[np.isfinite(p_xytheta)]
      Y = Y[np.isfinite(p_xytheta)]
      Theta = Theta[np.isfinite(p_xytheta)]
      p_xytheta = p_xytheta[np.isfinite(p_xytheta)]

      # Model distribution
      X1 = np.array(X, dtype=theano.config.floatX)
      Y1 = np.array(Y, dtype=theano.config.floatX)
      Theta1 = np.array(Theta, dtype=theano.config.floatX)

      Error = model_ttl([X1,Y1,Theta1],p_xytheta,[dx,dy,dtheta],cc,edge)
      print('Error',Error)

      # Yhat = tpdf(X1,edge.param_x[0],edge.param_x[1],edge.param_x[2]).eval()*tpdf(Y1,edge.param_y[0],edge.param_y[1],edge.param_y[2]).eval() * lpdf(Theta1,edge.param_theta[0],edge.param_theta[1]).eval()
      # fig = plt.figure()
      # ax = fig.add_subplot(111, projection='3d')
      # ax.scatter(X, Y, p_xytheta)
      # ax.scatter(X, Y, Yhat,c='r')
      # ax.set_xlabel('X Label')
      # ax.set_ylabel('Y Label')
      # ax.set_zlabel('P Label')
      # plt.show()
      self.save('TestImg3D.obj')

  def rigid_reg(self,edgeID):
    (cc, warp_matrix) = rigid_reg(self.edges[edgeID].node0.p_intensity,self.edges[edgeID].node1.p_intensity)
    #self.edges[edgeID].view(img2 = node1_transformed)
    return (cc,warp_matrix)

class MRI:

  def __init__(self,fname):
    self.info = nib.load('mri_small.nii')
    self.data = self.info.get_data()
    self.data = self.data/max(self.data.flatten())

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

  def reg_to_slice(self,IMG,dz=0,theta_x=0,theta_y=0):
    warpCoef = []
    warpList = []
    # per section
    with pymp.Parallel(4) as p:
      for i,img in enumerate(IMG.images):
        # x,y,z MRI
        x_mri = np.array([self.info.affine[0][0] * x + self.info.affine[0][3] for x in range(self.data.shape[0])])
        y_mri = np.array([self.info.affine[1][1] * y + self.info.affine[1][3] for y in range(self.data.shape[1])])
        z_mri = np.array([self.info.affine[2][2] * z + self.info.affine[2][3] for z in range(self.data.shape[2])])

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

        #interpolate results
        mri = interpn((x_mri, y_mri, z_mri), self.p_intensity, np.array([XXX, YYY, ZZZ]).T, bounds_error=False, fill_value=0)
        (cc, warp) = rigid_reg(mri.astype(np.float32),img.p_intensity)
        warpList.append(warp)
        warpCoef.append(cc)
        dy, dx, dtheta = warp[0][2], warp[1][2], np.arcsin(warp[0][1]) * 180 / np.pi
        print('max: ', dx, dy, dtheta, 'cc: ', cc)

    return (warpList, warpCoef)

  def reg_to_slice_TF(self,IMG,dz=0,theta_x=0,theta_y=0):
    params = {'load_file': '/Users/colin/Dropbox/__Atlas__/model_saves/model-regNET_257x265_21000',
              'save_file': 'regNET3',
              'save_interval': 1000,
              'batch_size': 32,
              'lr': .0001,  # Learning rate
              'rms_decay': 0.9,  # RMS Prop decay
              'rms_eps': 1e-8,  # RMS Prop epsilon
              'width': 265,
              'height': 257,
              'numParam': 3,
              'train': True}

    # go to training dir
    fsys.cd('/Users/colin/Dropbox/__Atlas__')

    ## initialize net
    net = AtlasNet(params)
    #test code remove
    #ideal = np.random.uniform(low=-10, high=10, size=(params['batch_size'], params['numParam']))

    ## create pairs of histology and sliced MRI
    batch = np.zeros(shape=(len(IMG.images), IMG.images[0].shape[1], IMG.images[0].shape[0], 2))
    # per section
    for i,img in enumerate(IMG.images):
      # x,y,z MRI
      x_mri = np.array([self.info.affine[0][0] * x + self.info.affine[0][3] for x in range(self.data.shape[0])])
      y_mri = np.array([self.info.affine[1][1] * y + self.info.affine[1][3] for y in range(self.data.shape[1])])
      z_mri = np.array([self.info.affine[2][2] * z + self.info.affine[2][3] for z in range(self.data.shape[2])])

      x = np.array([img.meta['MAT'][0][0] * x + img.meta['MAT'][0][3] for x in range(img.shape[0])])
      y = np.array([img.meta['MAT'][1][1] * y + img.meta['MAT'][1][3] for y in range(img.shape[1])])
      z = img.meta['MAT'][2][3] + dz
      XX, YY = np.atleast_2d(x, y)
      YY = YY.T

      affine_x = np.array([[1, 0, 0, 0],
                           [0, math.cos(math.radians(theta_x)), -1. * math.sin(math.radians(theta_x)), 0],
                           [0, math.sin(math.radians(theta_x)), math.cos(math.radians(theta_x)), 0],
                           [0, 0, 0, 1]])

      affine_y = np.array([[math.cos(math.radians(theta_y)), 0, math.sin(math.radians(theta_y)), 0],
                           [0, 1, 0, 0],
                           [-1. * math.sin(math.radians(theta_y)), 0, math.cos(math.radians(theta_y)), 0],
                           [0, 0, 0, 1]])
      MAT = np.dot(affine_x, affine_y)
      XXX = MAT[0][0] * XX + MAT[0][1] * YY + MAT[0][2] * z + MAT[0][3]
      YYY = MAT[1][0] * XX + MAT[1][1] * YY + MAT[1][2] * z + MAT[1][3]
      ZZZ = MAT[2][0] * XX + MAT[2][1] * YY + MAT[2][2] * z + MAT[2][3]
      mri = Img.from_array(interpn((x_mri, y_mri, z_mri), self.p_intensity, np.array([XXX, YYY, ZZZ]).T, bounds_error=False, fill_value=0))
      batch[i, :, :, :] = cv2.merge((np.rot90(mri.p_intensity), np.rot90(img.p_intensity)))

      # plt.figure()
      # merged = np.dstack(
      #   (np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 1]), np.array(batch[i, :, :, 1])))
      # plt.imshow(merged)
      # plt.show()

      #(cc, warp) = rigid_reg(mri.astype(np.float32),img.p_intensity)
      #warpList.append(warp)
      #warpCoef.append(cc)
      #dy, dx, dtheta = warp[0][2], warp[1][2], np.arcsin(warp[0][1]) * 180 / np.pi
      #print('max: ', dx, dy, dtheta, 'cc: ', cc)

    ## estimate transformations
    tformed, xytheta = net.run_batch(batch)
    for i in range(tformed.shape[0]):
      plt.figure()
      merged = np.dstack(
        (np.array(batch[i, :, :, 0]), np.array(batch[i, :, :, 1]), np.array(tformed[i][:][:][:]).squeeze()))
      plt.imshow(merged)
      plt.show()

    #test code
    #out = net.run_batch(batch)

    # avgCost = 100 * [np.inf]
    # for i in range(10000):
    #   cnt, cost, out = net.train(batch[np.random.randint(batch.shape[0], size=10),:,:,:])
    #   avgCost.append(cost)
    #   avgCost.pop(0)
    #   print('count: {}, cost: {}, avg_cost: {}, sanity: {}'.format(cnt, cost, np.mean(avgCost),
    #         np.mean(np.abs(out),axis=0)))
    #   if (params['save_file']):
    #     if cnt % params['save_interval'] == 0:
    #       net.save_ckpt('model_saves/model-' + params['save_file'] + "_" + str(params['width']) + 'x' + str(
    #         params['height']) + '_' + str(cnt))
    #       print('Model saved')
    #
    # # test code
    # out = net.run_batch(batch)
    return (tformed, xytheta)


#helper functions
# for p_intensity to ignore zero/background
def machineEpsilon(func=float):
     machine_epsilon = func(1)
     while func(1)+func(machine_epsilon) != func(1):
         machine_epsilon_last = machine_epsilon
         machine_epsilon = func(machine_epsilon) / func(2)
     return machine_epsilon_last
# only handle grayscale images
def rgb2gray(rgb):
  if len(rgb.shape)>2:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * b + 0.1140 * g
    return gray
  else:
    return rgb

def get_defaultMETA():
  affine = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  meta = {'MAT0':affine,'MAT':copy.deepcopy(affine),'fname':None,'parent':None,'Children':{}}
  return meta


def tpdf(value, lower, c, upper):
  # return ifelse(T.gt(T.concatenate([T.gt(lower,c), T.lt(upper,c), T.gt(lower,upper)]).sum(),0),T.zeros_like(value),T.switch(alltrue_elemwise([lower <= value, value < c]),
  #                 ((value - lower) / ((c - lower))),
  #                 T.switch(T.eq(value, c), (1),
  #                 T.switch(alltrue_elemwise([c < value, value <= upper]),
  #                 ((upper - value) / ((upper - c))), 0))))

  return T.switch(alltrue_elemwise([lower <= value, value < c]),
                  ((value - lower) / ((c - lower) +np.array(1e-32, dtype=theano.config.floatX))),
                  T.switch(T.eq(value, c), (1),
                  T.switch(alltrue_elemwise([c < value, value <= upper]),
                  ((upper - value) / ((upper - c) +np.array(1e-32, dtype=theano.config.floatX))), 0)))


def gpdf(sample, location, scale):
  divisor = 2 * scale ** 2
  exp_arg = -((sample - location) ** 2) / divisor
  return T.exp(exp_arg)


def lpdf(sample, location, scale):
  divisor = 2 * scale  # + epsilon,
  exp_arg = -T.abs_(sample - location) / divisor
  return T.exp(exp_arg)

def ecc(img0,img1):
  return np.dot(img0.flatten(), img1.flatten()) / np.linalg.norm(img0.flatten()) / np.linalg.norm(img1.flatten())

def model_ttl(locations, samples, centers, cc,edge):
  basic_model = pm.Model()
  # Xt = locations[0]
  # Yt = locations[1]
  # Thetat = locations[2]
  #
  # # Y1 = np.array(Y, dtype=theano.config.floatX)
  # Samplest = np.array(samples,dtype=theano.config.floatX)
  # center1 = np.array(centers[0],dtype=theano.config.floatX)
  # center2 = np.array(centers[1],dtype=theano.config.floatX)
  # center3 = np.array(centers[2],dtype=theano.config.floatX)
  # cct = np.array(cc,dtype=theano.config.floatX)

  with basic_model:
    # Priors for unknown model parameters
    l1 = pm.Normal('l1', mu=edge.param_x[0], sd=edge.node0.shape[0]/85)
    m1 = centers[0]
    u1 = pm.Normal('u1', mu=edge.param_x[2], sd=edge.node0.shape[0]/85)

    l2 = pm.Normal('l2', mu=edge.param_y[0], sd=edge.node0.shape[1]/85)
    m2 = centers[1]
    u2 = pm.Normal('u2', mu=edge.param_y[2], sd=edge.node0.shape[1]/85)

    m3 = centers[2]
    s3 = pm.HalfNormal('s3', sd=20)

    ## debug
    # fun = theano.function(
    #   [Xt, l1, m1, u1], tpdf,
    #   mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
    # )
    # fun2 = theano.function(
    #   [Yt, l2, m2, u2], tpdf,
    #   mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
    # )
    # p_x = fun(Xt, l1, m1, u1)
    # p_y = fun2(Yt, l2, m2, u2)
    ## end debug

    p_x = tpdf(locations[0], l1, m1, u1)
    p_y = tpdf(locations[1], l2, m2, u2)
    p_theta = lpdf(locations[2], m3, s3)

    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = cc * p_x * p_y * p_theta

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=samples)
    trace = pm.sample(2000, njobs=4)

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
  #print(Err)

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

def save_obj(var,fname):
  file_handler = open(fname, 'wb')
  pickle.dump(var, file_handler)

def load_obj(fname):
  filehandler = open(fname, 'rb')
  return pickle.load(filehandler)

def rigid_reg(fixed,moving):
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
  warp = np.eye(3)
  warp[0:2] = warp_matrix
  warp = np.linalg.inv(warp)[0:2]
  rows, cols = moving.shape
  node1_transformed = cv2.warpAffine(moving, warp, (cols, rows))
  cc = np.dot(node1_transformed.flatten(), fixed.flatten()) / np.linalg.norm(
    node1_transformed.flatten()) / np.linalg.norm(fixed.flatten())

  #combined = np.zeros(shape=(node1_transformed.shape[0],node1_transformed.shape[1],3))
  #combined[:,:,0]=node1_transformed/max(node1_transformed.flatten())
  #combined[:,:,1]=fixed
  #combined[:,:,2]=moving
  #plt.imshow(combined)
  #plt.show()
  #self.edges[edgeID].view(img2 = node1_transformed)
  return (cc,warp)

def affine_2d_to_3d(affine):
  new = np.eye(4)
  new[0:2,0:2] = affine[0:2,0:2]
  new[0:2,3] = affine[0:2,2]
  return new

def affine_make_full(affine):
  new = np.eye(max(affine.shape))
  new[0:affine.shape[0],0:affine.shape[1]] = affine
  return new


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype=R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def affine_get_params(R):
  assert (isRotationMatrix(R[0:3,0:3]))

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

def reverse_tform_order(theta,dx,dy):
  # calculate rotate -> translate from translate -> rotate
  dx1 = np.cos(np.deg2rad(theta))*dx - np.sin(np.deg2rad(theta))*dy
  dy1 = np.sin(np.deg2rad(theta))*dx + np.cos(np.deg2rad(theta))*dy

  # calculate translate -> rotate from rotate -> translate
  dx2 = (dx + (np.sin(np.deg2rad(theta))/np.cos(np.deg2rad(theta)))*dy) / (np.cos(np.deg2rad(theta)) + ((np.sin(np.deg2rad(theta)) ** 2) / np.cos(np.deg2rad(theta))))
  dy2 = (dy - (np.sin(np.deg2rad(theta))/np.cos(np.deg2rad(theta)))*dx) / (np.cos(np.deg2rad(theta)) + ((np.sin(np.deg2rad(theta)) ** 2) / np.cos(np.deg2rad(theta))))

  return [(dx1,dy1),(dx2,dy2)]