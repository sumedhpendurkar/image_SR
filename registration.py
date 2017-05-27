import cv2
import numpy as np
import sys
import scipy
from random import choice
from scipy import linalg
from scipy import ndimage
import math

def get_corners(img):
    #Conver to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    #get a array of probabilites kind of
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst,None)

    # determine where a corner or not change the 0.01 to determine threshold
    boolean_dst = dst > 0.01 * dst.max()
    
    coord_dst = []

    #convert boolean_dst into list of co-ordinates
    for i in xrange(len(boolean_dst)):
        for j in xrange(len(boolean_dst[i])):
            if boolean_dst[i][j]:
                coord_dst.append( (i,j) )

    #return both points in boolean form and the list
    return boolean_dst, coord_dst



def get_matches(img1, img2, dst_list1, dst_list2, window_size = 5):
    n = window_size
    point_list = []
    for point1 in dst_list1:
        max_ncc = -2
        correspond_point = None
        p1_x = point1[0]
        p1_y = point1[1]
        if p1_x - 2 >= 0 and p1_y - 2 >= 0:
            win1 = img1[p1_x - 2: p1_x + 3, p1_y - 2: p1_y + 3]
            win1 = np.pad(win1, ((0, n - win1.shape[0]), (0, n - win1.shape[1])), mode = 'constant', constant_values = 0)
        elif p1_y - 2 >= 0:
            win1 = img1[0: p1_x + 3, p1_y - 2: p1_y + 3]
            win1 = np.pad(win1, ((2 - p1_x, 0), (0, n - win1.shape[1])), mode = 'constant', constant_values = 0)
        elif p1_x - 2 >= 0:
            win1 = img1[p1_x - 2: p1_x + 3, 0: p1_y + 3]
            win1 = np.pad(win1, ((0, n - win1.shape[0]), (2 - p1_y, 0)), mode = 'constant', constant_values = 0)
        else:
            win1 = img1[0: p1_x + 3, 0: p1_y + 3]
            win1 = np.pad(win1, ((2 - p1_x, 0), (2 - p1_y, 0)), mode = 'constant', constant_values = 0)
        #print "win1 = ", win1
        win1_mean = img1.mean() * np.ones(win1.shape)
        win1_norm = ((win1 - win1_mean) * (win1 - win1_mean)).sum()

        for point2 in dst_list2:
            p2_x = point2[0]
            p2_y = point2[1]
            if p2_x - 2 >= 0 and p2_y - 2 >= 0:
                win2 = img2[p2_x - 2: p2_x + 3, p2_y - 2: p2_y + 3]
                win2 = np.pad(win2, ((0, n - win2.shape[0]), (0, n - win2.shape[1])), mode = 'constant', constant_values = 0)
            elif p2_y - 2 >= 0:
                win2 = img2[0: p2_x + 3, p2_y - 2: p2_y + 3]
                win2 = np.pad(win2, ((2 - p2_x, 0), (0, n - win2.shape[1])), mode = 'constant', constant_values = 0)
            elif p2_x - 2 >= 0:
                win2 = img2[p2_x - 2: p2_x + 3, 0: p2_y + 3]
                win2 = np.pad(win2, ((0, n - win2.shape[0]), (2 - p2_y, 0)), mode = 'constant', constant_values = 0)
            else:
                win2 = img2[0: p2_x + 3, 0: p2_y + 3]
                win2 = np.pad(win2, ((2 - p2_x, 0), (2 - p2_y, 0)), mode = 'constant', constant_values = 0)

            #print "win2 = ", win2
            win2_mean = img2.mean() * np.ones(win2.shape)
            win2_norm = ((win2 - win2_mean) * (win2 - win2_mean)).sum()
            denominator = math.sqrt(win1_norm * win2_norm)
            numerator = ((win2 - win2_mean) * (win1 - win1_mean)).sum()
            ncc = numerator/ denominator
            if ncc > max_ncc:
                max_ncc = ncc
                correspond_point = point2
        try:
            dst_list2.remove(correspond_point)
        except:
            pass
        point_list.append((point1, correspond_point))
    print dst_list2
    return point_list

def Haffine_from_points(fp,tp):
    """ find H, affine transformation, such that 
        tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError, 'number of points do not match'

    #condition points
    #-from points-
    m = scipy.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1))
    C1 = scipy.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = scipy.dot(C1,fp)

    #-to points-
    m = scipy.mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = scipy.dot(C2,tp)

    #conditioned points have mean zero, so translation is zero
    A = scipy.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = scipy.concatenate((scipy.dot(C,linalg.pinv(B)), np.zeros((2,1))), axis=1) 
    H = scipy.vstack((tmp2,[0,0,1]))

    #decondition
    H = scipy.dot(linalg.inv(C2),scipy.dot(H,C1))

    return H / H[2][2]



def ransac(im1, im2, points_list, iters = 10 , error = 10, good_model_num = 5):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''
    
    rows,cols = im1.shape

    model_error = 255
    model_H = None

    for i in range(iters):
        consensus_set = []
        points_list_temp = np.copy(points_list).tolist()
        # Randomly select 3 points
        for j in range(3):
            temp = choice(points_list_temp)
            consensus_set.append(temp)
            points_list_temp.remove(temp)
        
        # Calculate the homography matrix from the 3 points
        
        fp0 = []
        fp1 = []
        fp2 = []
        
        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:
        
            fp0.append(line[0][0])
            fp1.append(line[0][1])
            fp2.append(1)
            
            tp0.append(line[1][0])
            tp1.append(line[1][1])
            tp2.append(1)
            
        fp = np.array([fp0, fp1, fp2])
        tp = np.array([tp0, tp1, tp2])
        
        H = Haffine_from_points(fp, tp)
                            
        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = np.array([x1, y1, 1]).reshape(3,1)
            B = np.array([x2, y2, 1]).reshape(3,1)
            
            out = B - scipy.dot(H, A)
            dist_err = scipy.hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)            
            

        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0]
                x1, y1 = p[1]
                
                A = np.array([x0, y0, 1]).reshape(3,1)
                B = np.array([x1, y1, 1]).reshape(3,1)
                
                out = B - scipy.dot(H, A)
                dist_err = scipy.hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if (max(dists) < error) and (max(dists) < model_error):
                model_error = max(dists)
                model_H = H
                        
    return model_H
 
img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])
dst1, dist_cordinate1 = get_corners(img1)
dst2, dist_cordinate2 = get_corners(img2)


print len(dist_cordinate1)
print len(dist_cordinate2)
"""print dist_cordinate1
print "end"
print dist_cordinate2"""
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
point_list = get_matches(gray1, gray2, dist_cordinate1, dist_cordinate2)

#get matches by using NCC and store it in form [(x,y), (x',y')]


out = ransac(gray1, gray2, point_list)
print "H matrix", out
H_ = linalg.inv(out)
print H_
imtemp = scipy.ndimage.affine_transform(gray1, H_[:2, :2], [H_[0][2], H_[1][2]])
# Threshold for an optimal value, it may vary depending on the image.
#img1[dst1   >0.01*dst1.max()]=[0,0,255]
print (imtemp - gray2).tolist()
cv2.imshow('dst',gray2)
cv2.imshow('dsttmp', imtemp)
if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
