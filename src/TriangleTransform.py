import numpy as np
import cv2

class Transform:
    def getTriangleList(self, rect, pointList):
        subdiv = cv2.Subdiv2D(rect)
        for i in range(pointList.shape[0]):
            subdiv.insert((pointList[i][0], pointList[i][1]))
        return subdiv.getTriangleList()

    def __init__(self, src, dst, srcRect, dstRect):
        # Both src and dst are N x 2 numpy array, they are labeled by user
        [h, w] = src.shape
        self._label_num = h
        self._src = src
        self._dst = dst
        self._srcRect = srcRect
        self._dstRect = dstRect

    def insideRect(self, point, rect):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def insideTriangle(self, pt1, pt2, pt3, pt):
        # according to algorithm from 
        # http://mathworld.wolfram.com/TriangleInterior.html
        def det(u, v):
            return u[0] * v[1] - u[1] * v[0]
        v0 = pt1
        v1 = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        v2 = (pt3[0] - pt1[0], pt3[1] - pt1[1])
        a = (det(pt, v2) - det(v0, v2)) / det(v1, v2)
        b = - (det(pt, v1) - det(v0, v1)) / det(v1, v2)
        if a > 0 and b > 0 and a+b < 1:
            return True
        return False

    def getDstPoint(self, point, srcTriangle, dstTriangle):
        # according to algorithm from
        # https://devendrapratapyadav.github.io/FaceMorphing/
        A = np.mat(np.array(((srcTriangle[0], srcTriangle[2], srcTriangle[4]),
                      (srcTriangle[1], srcTriangle[3], srcTriangle[5]),
                      (1, 1, 1)
                    )))
        B = np.mat(np.array(((dstTriangle[0], dstTriangle[2], dstTriangle[4]),
                      (dstTriangle[1], dstTriangle[3], dstTriangle[5]),
                      (1, 1, 1)
                    )))
        T = B * np.linalg.inv(A)
        dst = T * np.mat(np.array((point[0], point[1], 1))).T
        return np.asarray(dst.T)[0]

    def getCorrepondingDst(self, srcTriangle):
        def getIndex(pt):
            for i, p in enumerate(self._src):
                if p[0] - pt[0] < 0.0001 and p[1] - pt[1] < 0.0001:
                    return i
            return None
        pt1 = (srcTriangle[0], srcTriangle[1])
        index1 = getIndex(pt1)
        if index1 == None:
            return None
        pt2 = (srcTriangle[2], srcTriangle[3])
        index2 = getIndex(pt2)
        if index2 == None:
            return None
        pt3 = (srcTriangle[4], srcTriangle[5])
        index3 = getIndex(pt3)
        if index3 == None:
            return None
        return (self._dst[index1][0], self._dst[index1][1],
                self._dst[index2][0], self._dst[index2][1],
                self._dst[index3][0], self._dst[index3][1])

    def Run(self, srcPoints):
        src_triangleList = self.getTriangleList(self._srcRect, self._src)
        dstPoints = srcPoints.copy()
        for i, t in enumerate(src_triangleList):
            pt1 = (t[0], t[1])
            if not self.insideRect(pt1, self._srcRect):
                continue
            pt2 = (t[2], t[3])
            if not self.insideRect(pt2, self._srcRect):
                continue
            pt3 = (t[4], t[5])
            if not self.insideRect(pt3, self._srcRect):
                continue
            srcTriangle = src_triangleList[i]
            dstTriangle = self.getCorrepondingDst(srcTriangle)
            if dstTriangle == None:
                continue
            xmax = max(pt1[0], pt2[0], pt3[0])
            xmin = min(pt1[0], pt2[0], pt3[0])
            ymax = max(pt1[1], pt2[1], pt3[1])
            ymin = min(pt1[1], pt2[1], pt3[1])
            for x in range(xmin.astype(int), xmax.astype(int) + 1):
                for y in range(ymin.astype(int), ymax.astype(int) + 1):
                    point = (x, y)
                    if self.insideTriangle(pt1, pt2, pt3, point):
                        [xt, yt, _]= self.getDstPoint(point, srcTriangle, dstTriangle)
                        dstPoints[y, x] = (xt, yt)
                        continue
        return dstPoints
                
