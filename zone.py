import cv2
import copy
from crop_image import crop_points

global frame_draw, frame_draw_clone, points

points = []

def draw_line(event, x, y, flags, param):
    global points, frame_draw, frame_draw_clone
    
    if event == cv2.EVENT_LBUTTONDOWN:  
        list_length = len(points)
        
        # only four points
        if list_length == 4:
            return
              
        points.append((x, y))
        
        cv2.circle(frame_draw, points[list_length], 5, (255,0,0), -1)
        cv2.line(frame_draw, points[list_length - 1], points[list_length], (0, 255, 0), 2)
        
        if list_length == 3:
            first_pt = points[0]
            last_pt = points[list_length]
            
            cv2.circle(frame_draw, last_pt, 5, (255,0,0), -1)
            cv2.line(frame_draw, first_pt, last_pt, (0, 255, 0), 2)

    elif event == cv2.EVENT_RBUTTONDOWN:  
        list_length = len(points)   
         
        if list_length == 0:
            return
        
        frame_draw = frame_draw_clone.copy()   
         
        points.remove(points[list_length - 1])
        
        for i in range (1, len(points)):
            cv2.circle(frame_draw, points[i - 1], 2, (255,0,0), 2)
            cv2.line(frame_draw, points[i - 1], points[i], (0,255,0), 2)
            
        if len(points) == 1:
            points.clear()
            
cap = cv2.VideoCapture('videos/video.mp4')
ratio = .7
ret, frame = cap.read()
frame_draw = copy.copy(cv2.resize(frame, (0, 0), None, ratio, ratio))
frame_draw_clone = copy.copy(frame_draw)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_line)

while True:
    cv2.imshow('frame', frame_draw)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):
        frame_draw = frame_draw_clone.copy()    
        points = []
        count = 0

    elif key == ord("e"):
        if len(points) != 4:
            cap.release()
            
        break

cv2.destroyAllWindows()

fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret is False:
        break
    
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
    fgmask = fgbg.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel)
    _, crop = crop_points(dilation, points)
    rect, frame_crop = crop_points(image, points)
    retvalbin, bins = cv2.threshold(crop, 220, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('crop', frame_crop)
    cv2.imshow('mask crop', crop)

    contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for countour in contours:
        (x, y, w, h) = cv2.boundingRect(countour)
        
        if cv2.contourArea(countour) < 1200:
            continue
        
        roi_y, roi_h, roi_x, roi_w = (
            rect[1] + y, 
            rect[1] + y + h,
            rect[0] + x,
            rect[0] + x + w
        )
    
        detection_crop = image[roi_y: roi_h, roi_x: roi_w]
        
        cv2.imshow('detection crop', detection_crop)
        
        cv2.putText(
            image,
            "X:" + str(roi_x) + " Y: " + str(roi_y),
            (roi_x, roi_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            1
        ) 
        cv2.rectangle(image, (roi_x, roi_y), (roi_w, roi_h), (255, 0, 0), 2)
    
        
    cv2.imshow('original', image)
        
    key = cv2.waitKey(30)
    
    if key == ord("e"):
        break

cv2.destroyAllWindows()
cap.release()
        

        
        
        