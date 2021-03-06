import cv2
import numpy as np
import sys
import scipy
from random import choice
from scipy import linalg
from scipy import ndimage
import math
import interpolate

def get_corners(img):
    """
    input : image
    output: 'R' values of all elements in matrix form, points above a threshhold
    """
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



def get_matches(img1, img2, dst_list1, dst_list2, threshold, window_size = 5):
    """
    Input: images, location of corners, size of window
    Output: list of matched points as lists
    Images are requried as NCC is implemented in spatial domain and uses intensity
    vectos
    """
    n = window_size

    #theres no need to recompute the windows in image 2 
    #store them and their means and norms
    point2_win = []
    point2_win_mean = []
    point2_win_norm = []
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
        win2_mean = win2.mean() * np.ones(win2.shape)
        win2_norm = ((win2 - win2_mean) * (win2 - win2_mean)).sum()
        point2_win.append(win2)
        point2_win_mean.append(win2_mean)
        point2_win_norm.append(win2_norm)

    #store the matched points in the list 
    point_list = []

    
    #take a point from list1 and all from list2; calculate ncc
    #if ncc is greater than a threshold for the highest match found store it
    #and remove that point
    for point1 in dst_list1:
        if len(dst_list2) == 0:
            break
        max_ncc = -2
        correspond_point = None
        p1_x = point1[0]
        p1_y = point1[1]

        #Create a window. Now assumed of 5 by 5
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
        win1_mean = win1.mean() * np.ones(win1.shape)
        win1_norm = ((win1 - win1_mean) * (win1 - win1_mean)).sum()

        index = 0
        while index <  len(point2_win):
            win2 = point2_win[index]
            win2_mean = point2_win_mean[index]
            win2_norm = point2_win_norm[index]
            denominator = math.sqrt(win1_norm * win2_norm)
            numerator = ((win2 - win2_mean) * (win1 - win1_mean)).sum()
            ncc = numerator/ denominator
            
            if ncc > max_ncc and ncc > threshold:
                max_ncc = ncc
                corresponding_index = index
            index+=1
        try:
            if max_ncc == -2:
                continue
            print max_ncc, (dst_list2[corresponding_index], point1)
            point_list.append((point1, dst_list2[corresponding_index]))
            del dst_list2[corresponding_index]
            del point2_win[corresponding_index]
            del point2_win_mean[corresponding_index]
            del point2_win_norm[corresponding_index]
        except:
            pass

    #print dst_list2
    return point_list


def ransac(im1, im2, points_list, iters = 10 , error = 10, good_model_num = 5):
    '''
        Input: images, points in form[(x,y),(x',y')] where those points are 
        corresponding points in image, number of interations of ransac,
        error in distance of point that can be tolerated, min number of points
        needed to say that array is good
        Output: returns transformation matrix H
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
        
        fsrc = []
        fdst = []
        for line in consensus_set: 
            fsrc.append(np.array([line[0][0], line[0][1]]))
            
            fdst.append(np.array([line[1][0], line[1][1]]))

        fp = np.float32(np.array(fsrc))
        tp = np.float32(np.array(fdst))
        
        H = cv2.getAffineTransform(fp, tp)
                            
        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = np.array([x1, y1, 1]).reshape(3,1)
            B = np.array([x2, y2]).reshape(2,1)
            
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
                B = np.array([x1, y1]).reshape(2,1)
                
                out = B - scipy.dot(H, A)
                dist_err = scipy.hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if (max(dists) < error) and (max(dists) < model_error):
                model_error = max(dists)
                model_H = H
                        
    return model_H

if __name__ == '__main__':
    debug = True
    try:
        img1 = cv2.imread(sys.argv[1])
        img2 = cv2.imread(sys.argv[2])
    except:
        print "usage python registration.py <img1> <img2> <threshold_for_ncc>"
    
    #get locations of corners
    dst1, dist_cordinate1 = get_corners(img1)
    dst2, dist_cordinate2 = get_corners(img2)

    if debug:
        print len(dist_cordinate1)
        print len(dist_cordinate2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #get corresponding points
    point_list = get_matches(gray1, gray2, dist_cordinate1, dist_cordinate2, float(sys.argv[3]))
    print point_list
#get matches by using NCC and store it in form [(x,y), (x',y')]


    out = ransac(gray1, gray2, point_list)
    print "H matrix", out
  
    upscaled_gray1 = interpolate.get_upscaled(gray1)

    # as we have H calculate predicated second image from first image
    imtemp = cv2.warpAffine(np.float32(upscaled_gray1), out, (2 * gray1.shape[1], 2 * gray1.shape[0]))
    
    final_img = interpolate.merge_images(upscaled_gray1, np.uint8(imtemp))
    #cv2.imshow('merged', np.uint8(final_img))
    cv2.imshow(sys.argv[2][:-4]+'merged.png', np.uint8(final_img))
    final_img = interpolate.image_interpolate2(final_img)
    #cv2.imshow('interpolated', np.uint8(final_img))
    cv2.imshow(sys.argv[2][:-4]+ 'interpolated.png', np.uint8(final_img))
    if debug:
        print "gray2 = ", gray2
        print  "imtemp= ", imtemp
    
    imtemp = np.uint8(imtemp)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destoryAllWindows()
#cv2.imwrite(sys.argv[2] + sys.argv[3] + "registered.png", imtemp)
