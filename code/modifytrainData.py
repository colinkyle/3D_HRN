import cv2
import fsys
import numpy as np
from imtools import Img
import tifffile
import matplotlib.pyplot as plt

fsys.cd('D:/__Atlas__/Train_nonbrain')
imgs = fsys.file('*.tiff')

def ncc(x,y):
    cc = np.dot(x.flatten(), y.flatten()) / np.linalg.norm(
        x.flatten()) / np.linalg.norm(y.flatten())
    return cc
for imname in imgs:
    with tifffile.TiffFile(imname) as tif:
      img = tif.asarray()
    img[img == 255.0] = 0
    img = img / 255.0

    # ch1 = img[:, :, 0]
    # ch2 = img[:, :, 1]
    # ch3 = img[:, :, 2]
    # y = ncc(ch1,ch2)
    # t = ncc(ch2,ch3)
    # m = ncc(ch1,ch3)
    # if t == max([y,t,m]):
    #     img = cv2.merge((ch2,ch3,ch1))
    # elif m == max([y,t,m]):
    #     img = cv2.merge((ch1, ch3, ch2))
    #img = cv2.merge((ch1, ch3, ch2))
    img = cv2.merge((Img.from_array(img[:, :, 0]).p_intensity,
                     Img.from_array(img[:, :, 1]).p_intensity,
                     Img.from_array(img[:, :, 2]).p_intensity))
    tifffile.imsave(imname, img)
    #cv2.imwrite(imname,img)

