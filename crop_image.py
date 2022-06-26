import numpy as np
import cv2

def crop_center(img,cropx,cropy): #
    y,x, channels = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
  
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_points(img, points):
    mask = np.zeros(img.shape[0:2], dtype = np.uint8)
    
    points = np.array([points])
    
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_8)
    
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points)
    
    return rect, res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
