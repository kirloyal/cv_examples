import numpy as np
import cv2
import scipy
import io
import os 
import matplotlib.pyplot as plt

camera_matrix0 =  np.loadtxt( "sample_imgs/epi0_calib.txt" )
camera_matrix1 =  np.loadtxt( "sample_imgs/epi1_calib.txt" )

img0 = cv2.imread("sample_imgs/epi0.jpg")
img1 = cv2.imread("sample_imgs/epi1.jpg")

C0 = -np.linalg.inv(camera_matrix0[:,:3]) @ camera_matrix0[:,3]
C0 = np.append(C0,1)
C1 = -np.linalg.inv(camera_matrix1[:,:3]) @ camera_matrix1[:,3]
C1 = np.append(C1,1)

e0 = camera_matrix0 @ C1
e0 = e0[:2]/e0[2] 
e1 = camera_matrix1 @ C0
e1 = e1[:2]/e1[2] 
cv2.circle(img0, tuple(e0.astype(int).tolist()), 5, (0,0,0),2)
cv2.circle(img1, tuple(e1.astype(int).tolist()), 5, (0,0,0),2)

cv2.imwrite('results/epi0.jpg', img0)
cv2.imwrite('results/epi1.jpg', img1)

