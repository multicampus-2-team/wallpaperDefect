import os
import cv2 as cv
import numpy as np

for i in os.listdir('../../open/train/가구수정/'):

    path = '../../open/train/가구수정/'+i

    img_array = np.fromfile(path, np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)

    #threshold, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    #cv.imshow( "image", thresh)

    canny = cv.Canny(img, 125, 175)
    #cv.imshow('Canny Edges', canny)
    #cv.waitKey(0)

    dilated = cv.dilate(canny, (7, 7), iterations=3)
    cv.imshow('Dilated', dilated)
    cv.waitKey(0)

    eroded = cv.erode(dilated, (7, 7), iterations=3)
    cv.imshow('Eroded', eroded)
    cv.waitKey(0)