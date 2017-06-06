import cv2
import scipy
from scipy import linalg
import numpy as np
import sys
from scipy import interpolate as intp
import math

def get_upscaled(src, ratio = 2):
    rows, cols = src.shape
    img = np.zeros((rows*ratio, cols*ratio), dtype = np.uint8)
    for x in range(0, rows):
        for y in range(0, cols):
            img[x*ratio][y*ratio] = src[x][y]
    return img

def merge_images(img1, img2):
    r, c = img1.shape
    img = np.zeros(img1.shape)
    for x in xrange(r):
        for y in xrange(c):
            if img1[x][y] > 0 and img2[x][y] > 0:
                img[x][y] = (img1[x][y]+img2[x][y]) / 2
            else:
                img[x][y] = max(img1[x][y], img2[x][y])
    return img


def image_interpolate2(img):
    r, c = img.shape
    is_valid = lambda x,y: x >=0 and x<r and y >= 0 and y < c
    is_non_empty = lambda x,y: img[x][y] > 0
    add_point = lambda x,y : pointx.append(x) or pointy.append(y) or intensity.append(img[x][y])
    for x in xrange(r):
        for y in xrange(c):
            if img[x][y] == 0:
                #interpolate
                incrementor = 1
                pointx = []
                pointy = []
                intensity = []
                while incrementor <= 10 and len(pointx) != 3:
                    split = 0
                    while split <= incrementor:
                        spt = incrementor - split
                        if is_valid(x-spt,y - split) and is_non_empty(x-spt,y - split):
                            add_point(x - spt, y - split)
                            if len(pointx) == 3:
                                break
                        elif is_valid(x+spt, y - split) and is_non_empty(x+spt,y - split):
                            add_point(x+spt, y - split)
                            if len(pointx) == 3:
                                break
                        elif is_valid(x - spt, y + split) and is_non_empty(x - spt, y + split):
                            #add point
                            add_point(x - spt, y + split)
                            if len(pointx) == 3:
                                break
                        elif is_valid(x + spt, y + split) and is_non_empty(x + spt, y + split):
                            #add_point
                            add_point(x + spt, y + split)
                            if len(pointx) == 3:
                                break
                        split+=1
                    incrementor+=1
                distances = map(lambda x1,y1: math.sqrt((x1-x)**2 + (y1-y) **2), pointx,pointy)
                #print distances
                #print x,y
                #print pointx
                #print pointy
                weighted_sum = sum(map(lambda x,y: x/y, intensity, distances))
                img[x][y] = weighted_sum / sum(map(lambda x: 1/x, distances))
                if True or img[x][y] > 10:
                    continue
                print '---start---'
                print x,y
                print pointx
                print pointy
                print intensity
                print img[x][y]
                print '---end---'
                #bilinear = intp.interp2d(pointx, pointy, intensity, kind = 'linear')
                #img[x][y]  = bilinear(x,y)
                """if img[x][y] >= 255:
                    print "--start--"
                    print x,y
                    print pointx
                    print pointy
                    print intensity
                    print "--end--"
                """
    return img


if __name__ == '__main__':
    img1_color = cv2.imread(sys.argv[1])
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)

    img2 = get_upscaled(img1)
    cv2.imshow('upscaled', img2)

    cv2.imshow('original', img1)
    print np.count_nonzero(img2)
    img3 = image_interpolate2(img2)
    print np.count_nonzero(img3)
    cv2.imshow('interpolated', np.uint8(img3))

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
