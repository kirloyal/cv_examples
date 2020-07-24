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
fmt_res = "results/tri{}.jpg"
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


def get_triangulation_exact(xys, camera_matrixs):
    """
    param xys : list of points in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return X,s : the point in 3d space and corresponding singular value
    """
    A = np.empty((0,4))
    for x, camera_matrix in zip(xys, camera_matrixs):
        A = np.append(A, x[:2].reshape(2,1) @ camera_matrix[2:3] - camera_matrix[:2], axis = 0)
    
    A /= np.linalg.norm(A[:,:3], axis=-1, keepdims=True)
    xyz, res, _, _ =  np.linalg.lstsq(A[:,:3], -A[:,-1:], rcond=None)
    return xyz[:,0], res
    

def get_triangulation_approximate(xys, camera_matrixs):
    """
    param xys : list of points in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return X,s : the point in 3d space and corresponding singular value
    """
    A = np.empty((0,4))
    for x, camera_matrix in zip(xys, camera_matrixs):
        A = np.append(A, x[:2].reshape(2,1) @ camera_matrix[2:3] - camera_matrix[:2], axis = 0)
    
    _,s,vh = np.linalg.svd(A)
    X = vh[-1]
    X = X/X[-1]

    return X[:-1], s[-1]
    
def get_triangulation_cv2(xys, camera_matrixs):
    """
    param xys : list of points in cameras
    param camera_matrixs : np.array{ v x 3 x 4 }
    return X,s : the point in 3d space and corresponding singular value
    """
    xyzs = []
    for i in range(len(camera_matrixs)-1):
        for j in range(i+1, len(camera_matrixs)):
            xyzs.append(cv2.triangulatePoints(camera_matrixs[i], camera_matrixs[j], xys[i].astype(float), xys[j].astype(float)))
    xyzs = np.array(xyzs, dtype=float)[...,0]
    xyzs /= xyzs[:,-1:]
    xyz = xyzs.mean(0)
    return xyz[:3], None

pt_names = [dfs[i].loc[:,0].tolist() for i in range(n_tgt)]
pt_names_all = list(functools.reduce(lambda x,y: set(x).union(set(y)), pt_names))
pt_xys = [dfs[i].loc[:,1:].to_numpy() for i in range(n_tgt)]
colors = {}
xyzs_approx = {}
xyzs_exact = {}
xyzs_cv2 = {}
for pt_name in pt_names_all:
    colors[pt_name] = np.random.randint(256, size=3)
    xys_valid = []
    camera_matrixs_valid = []
    for i in range(n_tgt):
        if pt_name in pt_names[i]:
            idx = pt_names[i].index(pt_name)
            xys_valid.append(pt_xys[i][idx])
            camera_matrixs_valid.append(camera_matrixs[i])
    if len(camera_matrixs_valid) >= 2:
        xyz, err = get_triangulation_exact(np.vstack(xys_valid), np.array(camera_matrixs_valid))
        xyzs_exact[pt_name] = xyz, err
        xyz, err = get_triangulation_approximate(np.vstack(xys_valid), np.array(camera_matrixs_valid))
        xyzs_approx[pt_name] = xyz, err
        xyz, err = get_triangulation_cv2(np.vstack(xys_valid), np.array(camera_matrixs_valid))
        xyzs_cv2[pt_name] = xyz, err

for pt_name in xyzs_exact.keys():
    print(pt_name, np.linalg.norm(xyzs_exact[pt_name][0] - xyzs_approx[pt_name][0]), np.linalg.norm(xyzs_exact[pt_name][0] - xyzs_cv2[pt_name][0]) )


for i in range(n_tgt):
    pt_names_i = dfs[i].loc[:,0].tolist()
    pts = dfs[i].loc[:,1:].to_numpy()
    for k in range(len(pts)):
        color = colors[pt_names_i[k]]
        cv2.circle(imgs[i], tuple(pts[k].astype(int).tolist()), 5, (0,0,0), -1)
        for j in range(n_tgt):
            if i == j:
                continue
            e = camera_matrixs[i] @ Cs[j]
            e = e/e[2] 
            cv2.circle(imgs[i], tuple(e[:2].astype(int).tolist()), 5, (0, 0, 0), -1)
            cv2.circle(imgs[i], tuple(e[:2].astype(int).tolist()), 3, (150,150,150), -1)
        for pt_name in xyzs_exact.keys():
            if pt_name in pt_names[i]:
                xyz = xyzs_approx[pt_name]
                xy1 = camera_matrixs[i] @ np.append(xyz[0],1)
                xy1 = xy1/xy1[-1] 
                xy = xy1[:-1]
                cv2.circle(imgs[i], tuple(xy.astype(int).tolist()), 3, (0, 0, 255), -1)
                cv2.circle(imgs[i], tuple(xy.astype(int).tolist()), 1, colors[pt_name].tolist(), -1)

                xyz = xyzs_exact[pt_name]
                xy1 = camera_matrixs[i] @ np.append(xyz[0],1)
                xy1 = xy1/xy1[-1] 
                xy = xy1[:-1]
                cv2.circle(imgs[i], tuple(xy.astype(int).tolist()), 3, (255, 0, 0), -1)
                cv2.circle(imgs[i], tuple(xy.astype(int).tolist()), 1, colors[pt_name].tolist(), -1)
                            
    cv2.imwrite(fmt_res.format(tgts[i]), imgs[i])

