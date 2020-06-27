import numpy as np
import cv2
import scipy
import io
import os 
import matplotlib.pyplot as plt
import pandas as pd

camera_matrix0 =  np.loadtxt( "sample_imgs/epi0_calib.txt" )
camera_matrix1 =  np.loadtxt( "sample_imgs/epi1_calib.txt" )

img0 = cv2.imread("sample_imgs/epi0.jpg")
img1 = cv2.imread("sample_imgs/epi1.jpg")
h0, w0, _ = img0.shape
h1, w1, _ = img1.shape
pd0 = pd.read_csv("sample_imgs/epi0_points.txt", delimiter = ' ', header=None)
pd1 = pd.read_csv("sample_imgs/epi1_points.txt", delimiter = ' ', header=None)

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

def get_cross(A):
    return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
F = get_cross(camera_matrix1 @ C0) @ camera_matrix1 @ np.linalg.pinv(camera_matrix0)

# x1 @ F 
pts0 = pd0.loc[:,1:].to_numpy()
for i in range(len(pts0)):
    pt0 = pts0[i]
    cv2.circle(img0, tuple(pts0[i].astype(int).tolist()), 5, (0,0,0),2)
    l1 = np.append(pt0,1) @ F.T
    cv2.line(img1, (0, int(-l1[2]/l1[1])), (w1 , int(-(l1[0]*w1+l1[2])/l1[1])), (255,0,0), 1)


pts1 = pd1.loc[:,1:].to_numpy()
for i in range(len(pts1)):
    pt1 = pts1[i]
    cv2.circle(img1, tuple(pts1[i].astype(int).tolist()), 5, (0,0,0),2)
    l0 = np.append(pt1,1) @ F
    cv2.line(img0, (0, int(-l0[2]/l0[1])), (w0 , int(-(l0[0]*w0+l0[2])/l0[1])), (255,0,0), 1)

cv2.imwrite('results/epi0.jpg', img0)
cv2.imwrite('results/epi1.jpg', img1)

