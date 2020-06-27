import numpy as np
import cv2
import scipy
import io
import os 
import matplotlib.pyplot as plt
import pandas as pd


fmt_img = "sample_imgs/epi{}.jpg"
fmt_calib = "sample_imgs/epi{}_calib.txt"
fmt_pts = "sample_imgs/epi{}_points.txt"
fmt_res = "results/epi{}.jpg"
tgts = [1, 2]

img0 = cv2.imread(fmt_img.format(tgts[0]))
img1 = cv2.imread(fmt_img.format(tgts[1]))
h0, w0, _ = img0.shape
h1, w1, _ = img1.shape


camera_matrix0 =  np.loadtxt(fmt_calib.format(tgts[0]))
camera_matrix1 =  np.loadtxt(fmt_calib.format(tgts[1]))

pd0 = pd.read_csv(fmt_pts.format(tgts[0]), delimiter = ' ', header=None)
pd1 = pd.read_csv(fmt_pts.format(tgts[1]), delimiter = ' ', header=None)

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

pt_names0 = pd0.loc[:,0].tolist()
pt_names1 = pd1.loc[:,0].tolist()

pt_names_all = list(set(pt_names0).union(set(pt_names1)))
colors = {}
for pt_name in pt_names_all:
    colors[pt_name] = np.random.randint(256, size=3)

pts0 = pd0.loc[:,1:].to_numpy()
for i in range(len(pts0)):
    pt_name0 = pt_names0[i]
    color = colors[pt_name0]
    pt0 = pts0[i]
    cv2.circle(img0, tuple(pts0[i].astype(int).tolist()), 5, color.tolist(), 2)
    if pt_name0 in pt_names1:
        l1 = np.append(pt0,1) @ F.T
        cv2.line(img1, (0, int(-l1[2]/l1[1])), (w1 , int(-(l1[0]*w1+l1[2])/l1[1])), color.tolist(), 1)


pts1 = pd1.loc[:,1:].to_numpy()
for i in range(len(pts1)):
    pt_name1 = pt_names1[i]
    color = colors[pt_name1]
    pt1 = pts1[i]
    cv2.circle(img1, tuple(pts1[i].astype(int).tolist()), 5, color.tolist(),2)
    if pt_name1 in pt_names0:
        l0 = np.append(pt1,1) @ F
        cv2.line(img0, (0, int(-l0[2]/l0[1])), (w0 , int(-(l0[0]*w0+l0[2])/l0[1])), color.tolist(), 1)


cv2.imwrite(fmt_res.format(tgts[0]), img0)
cv2.imwrite(fmt_res.format(tgts[1]), img1)

