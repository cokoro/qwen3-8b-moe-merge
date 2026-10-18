import numpy as np
import json
import os
import sys
import scipy.linalg

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_npy(path, data):
    np.save(path, data)

def load_npz(path):
    return np.load(path)

def vee(mat):
    return np.array([mat[2,1], mat[0,2], mat[1,0]])

def cross(x, y):
    return np.array([x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]])

def rot_x(theta):
    return np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

def rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

def trans_x(a):
    return np.array([[1, 0, 0, a],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

def trans_z(d):
    return np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, d],
                    [0, 0, 0, 1]])

def compute_T(q, a, alpha, d, theta_offset):
    n = len(q)
    T = np.eye(4)
    for i in range(n):
        theta = q[i] + theta_offset[i]
        T_i = trans_z(d[i]) @ rot_z(theta) @ trans_x(a[i]) @ rot_x(alpha[i])
        T = T @ T_i
    return T

def compute_R(q, a, alpha, d, theta_offset):
    n = len(q)
    R = np.eye(3)
    for i in range(n):
        theta = q[i] + theta_offset[i]
        R_i = rot_z(theta) @ rot_x(alpha[i])
        R = np.linalg.multi_dot([R_i, R])
    return R

def compute_p_com(q, a, alpha, d, theta_offset, com):