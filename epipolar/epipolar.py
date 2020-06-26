import numpy as np
import cv2
import scipy
import io
import os 
import matplotlib.pyplot as plt

camera_matrix0 =  np.loadtxt( "sample_imgs/epi0_calib.txt" )
camera_matrix1 =  np.loadtxt( "sample_imgs/epi1_calib.txt" )

img0 = cv2.cvtColor(cv2.imread("sample_imgs/epi0.jpg"), cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(cv2.imread("sample_imgs/epi1.jpg"), cv2.COLOR_BGR2RGB)


# cv2.imwrite('data/dst/lena_opencv_red.jpg', img0)
# cv2.imwrite('results/epi1.jpg', img0)
