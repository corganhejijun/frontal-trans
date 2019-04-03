import numpy as np
import functools
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def Error_Rigid(x0, src, dst, weight_map, p_star, q_star, buf, buf_map):
    label_num = src.shape[0]
    a_cos = np.cos(x0)
    a_sin = np.sin(x0)

    for i in range(label_num):
        p = src[i, :]
        q = dst[i, :]
        p_bar = -p_star + p
        q_bar = -q_star + q
        buf[:, 0] = p_bar[:, 0] * a_cos + p_bar[:, 1] * a_sin
        buf[:, 1] = p_bar[:, 0] * -a_sin + p_bar[:, 1] * a_cos
        buf -= q_bar

        buf_map[:, i] = np.linalg.norm(buf, axis=1) * weight_map[:, i]

    error = np.sum(buf_map, axis=1)

    return error

class MovingLSQ:
    def __init__(self, src, dst):
        #
        # Both src and dst are N x 2 numpy array, they are labeled by user
        #  
        [h, w] = src.shape
        self._label_num = h
        self._src = src
        self._dst = dst
    
    def Run_Rigid(self, srcPts, alpha = 1):
        npoints = srcPts.shape[0]
        label_num = self._label_num
        weight = np.zeros([npoints, label_num], np.float)

        for i in range(label_num):
            label = self._src[i, :]
            D = np.linalg.norm( srcPts - label , axis = 1)
            D[D == 0] += 0.000001
            weight[:, i] = D[:]
            #weight += 0.00001 # prevent zero
        weight = 1 / (weight**(2*alpha))

        # weigth is npoints x label_num
        jacobian = lil_matrix((npoints, npoints), dtype=int)
        idx = np.arange(npoints)
        jacobian[idx, idx] = 1
        x0 = np.zeros(npoints)

        p_star = np.zeros([npoints, 2], np.float)
        q_star = np.zeros([npoints, 2], np.float)

        p_tmp = np.zeros([npoints, 2], np.float)
        q_tmp = np.zeros([npoints, 2], np.float)
        
        for i in range(label_num):
            p = self._src[i, :]
            q = self._dst[i, :]

            p_tmp[:, 0] += weight[:, i] * p[0]
            p_tmp[:, 1] += weight[:, i] * p[1]
            q_tmp[:, 0] += weight[:, i] * q[0]
            q_tmp[:, 1] += weight[:, i] * q[1]
        p_star[:, 0] = p_tmp[:, 0] / np.sum(weight, axis=1)
        p_star[:, 1] = p_tmp[:, 1] / np.sum(weight, axis=1)
        q_star[:, 0] = q_tmp[:, 0] / np.sum(weight, axis=1)
        q_star[:, 1] = q_tmp[:, 1] / np.sum(weight, axis=1)

        buf = np.zeros([npoints, 2], np.float)
        buf_map = np.zeros([npoints, label_num], np.float)

        lower_bound = np.zeros(npoints) - np.pi
        upper_bound = np.zeros(npoints) + np.pi
        #print Error_2D_Rigid(x0, self._src, self._dst, weight, p_star, q_star, buf, buf_map)
        #exit()
        result = least_squares(Error_Rigid, x0, verbose=0, jac_sparsity = jacobian, bounds=(lower_bound, upper_bound),
                args=(self._src, self._dst, weight, p_star, q_star, buf, buf_map))
        x_cos = np.cos(result['x'])
        x_sin = np.sin(result['x'])
        buf[:, 0] = p_star[:, 0] * x_cos + p_star[:, 1] * x_sin
        buf[:, 1] = p_star[:, 0] * -x_sin + p_star[:, 1] * x_cos
        T = q_star - buf
        buf[:, 0] = srcPts[:, 0] * x_cos + srcPts[:, 1] * x_sin
        buf[:, 1] = srcPts[:, 0] * -x_sin + srcPts[:, 1] * x_cos
        return buf + T
