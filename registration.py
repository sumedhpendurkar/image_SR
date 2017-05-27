import cv2
import numpy as np
import sys
def get_corners():
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

filename = sys.argv[1]
img = cv2.imread(filename)

dst, _ = get_corners()
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
