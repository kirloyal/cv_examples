import numpy as np
import cv2
import scipy
import io
import os 
import functools
import matplotlib.pyplot as plt
import pandas as pd


fmt_img = "sample_imgs/epi{}.jpg"
fmt_calib = "sample_imgs/epi{}_calib.txt"
fmt_pts = "sample_imgs/epi{}_points.txt"
fmt_res = "results/epi{}.jpg"
tgts = [0, 1, 2]
n_tgt = len(tgts)

imgs = []
hs, ws = [], []
camera_matrixs = []
dfs = []
Cs = []
for tgt in tgts:
    img = cv2.imread(fmt_img.format(tgt))
    imgs.append(img)
    h, w, _ = img.shape
    hs.append(h)
    ws.append(w)
    camera_matrix = np.loadtxt(fmt_calib.format(tgt))    
    camera_matrixs.append(camera_matrix)
    dfs.append(pd.read_csv(fmt_pts.format(tgt), delimiter = ' ', header=None))
    C = -np.linalg.inv(camera_matrix[:,:3]) @ camera_matrix[:,3]
    C = np.append(C,1)
    Cs.append(C)

es = {}
Fs = {}
def get_cross(A):
    return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])

for i in range(n_tgt):
    for j in range(n_tgt):
        if i == j:
            continue
        e = camera_matrixs[j] @ Cs[i]
        e = e/e[2] 
        F = get_cross(e) @ camera_matrixs[j] @ np.linalg.pinv(camera_matrixs[i])
        Fs[i,j] = F
        es[j,i] = e[:2]

pt_names = [dfs[i].loc[:,0].tolist() for i in range(n_tgt)]
pt_names_all = list(functools.reduce(lambda x,y: set(x).union(set(y)), pt_names))
colors = {}
for pt_name in pt_names_all:
    colors[pt_name] = np.random.randint(256, size=3)


for i in range(n_tgt):
    pt_names_i = dfs[i].loc[:,0].tolist()
    pts = dfs[i].loc[:,1:].to_numpy()
    for k in range(len(pts)):
        color = colors[pt_names_i[k]]
        cv2.circle(imgs[i], tuple(pts[k].astype(int).tolist()), 5, color.tolist(), 2)
    for j in range(n_tgt):
        if i == j:
            continue
        cv2.circle(imgs[i], tuple(es[i,j].astype(int).tolist()), 5, (150,150,150),3)
        pts = dfs[j].loc[:,1:].to_numpy()
        pt_names_j = dfs[j].loc[:,0].tolist()
        for k in range(len(pts)):
            pt_name = pt_names_j[k]
            if pt_name in pt_names_i:
                color = colors[pt_name]
                pt = pts[k]
                l = np.append(pt,1) @ Fs[i,j]
                cv2.line(imgs[i], (0, int(-l[2]/l[1])), (ws[i] , int(-(l[0]*ws[i]+l[2])/l[1])), color.tolist(), 1)
        
    cv2.imwrite(fmt_res.format(tgts[i]), imgs[i])
