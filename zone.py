import cv2
import copy
from crop_image import crop_points

global frame_draw, frame_draw_clone, points, line

points = []
line = []

def draw_line(event, x, y, flags, param):
    global line, frame_draw, frame_draw_clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        line_length = len(line)
        
        if line_length == 2:
            return
              
        line.append((x, y))
        
        cv2.circle(frame_draw, line[line_length], 5, (255,0,0), -1)
        cv2.circle(frame_draw, line[line_length - 1], 5, (255,0,0), -1)
        cv2.line(frame_draw, line[line_length - 1], line[line_length], (0,0,255), 2)    
        
    elif event == cv2.EVENT_RBUTTONDOWN: 
        line_length = len(line)   
         
        if line_length == 0:
            return
        
        frame_draw = frame_draw_clone.copy()   
         
        line.remove(line[line_length - 1])
        
        for i in range (1, len(line)):
            cv2.circle(frame_draw, line[i - 1], 2, (255,0,0), 2)
            cv2.circle(frame_draw, line[i], 2, (255,0,0), 2)
            cv2.line(frame_draw, line[i - 1], line[i], (0,255,0), 2)
            
        if len(line) == 1:
            line.clear()
         
def draw_points(event, x, y, flags, param):
    global points, frame_draw, frame_draw_clone
    
    if event == cv2.EVENT_LBUTTONDOWN:  
        points_length = len(points)
        
        if points_length == 4:
            return
              
        points.append((x, y))
        
        cv2.circle(frame_draw, points[points_length], 5, (255,0,0), -1)
        cv2.circle(frame_draw, points[points_length - 1], 5, (255,0,0), -1)
        cv2.line(frame_draw, points[points_length - 1], points[points_length], (0,255,0), 2)
        
        if points_length == 3:
            first_pt = points[0]
            last_pt = points[3]
            
            cv2.circle(frame_draw, last_pt, 5, (255,0,0), -1)
            cv2.line(frame_draw, first_pt, last_pt, (0, 255, 0), 2)

    elif event == cv2.EVENT_RBUTTONDOWN:  
        points_length = len(points)   
         
        if points_length == 0:
            return
        
        frame_draw = frame_draw_clone.copy()   
         
        points.remove(points[points_length - 1])
        
        for i in range (1, len(points)):
            cv2.circle(frame_draw, points[i - 1], 2, (255,0,0), 2)
            cv2.circle(frame_draw, points[i], 2, (255,0,0), 2)
            cv2.line(frame_draw, points[i - 1], points[i], (0,255,0), 2)
            
        if len(points) == 1:
            points.clear()
      
ratio = .7           
cap = cv2.VideoCapture('videos/video.mp4')
ret, frame = cap.read()
frame_draw = copy.copy(cv2.resize(frame, (0, 0), None, ratio, ratio))
frame_draw_clone = copy.copy(frame_draw)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_points)

while True:
    cv2.imshow('frame', frame_draw)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):
        frame_draw = frame_draw_clone.copy()    
        points = []

    elif key == ord("p"):
        if len(points) != 4:
            cap.release()     
        else:
            _, frame_crop = crop_points(frame_draw_clone, points)
            
            frame_draw_clone = frame_crop
            frame_draw = frame_draw_clone.copy()
            
            cv2.setMouseCallback('frame', draw_line)    
         
    elif key == ord("l"):
        if len(points) != 4 or len(line) != 2:
            cap.release() 
             
        break
 

cv2.destroyAllWindows()

fgbg = cv2.createBackgroundSubtractorMOG2()
min_contour_width = 80
min_contour_height = 50
offset = 7

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret is False:
        break
    
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel)
    _, crop = crop_points(dilation, points)
    rect, frame_crop = crop_points(image, points)
    retvalbin, bins = cv2.threshold(crop, 220, 255, cv2.THRESH_BINARY)

    for i in range (1, len(line)):
        cv2.circle(frame_crop, line[i - 1], 2, (255,0,0), 2)
        cv2.circle(frame_crop, line[i], 2, (255,0,0), 2)
        cv2.line(frame_crop, line[i - 1], line[i], (0,0,255), 2)

    contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    
    cv2.drawContours(frame_crop, hull, -1, (0, 255, 0), 3)
    cv2.imshow('crop', frame_crop)
    cv2.imshow('mask crop', crop)
         
    for countour in contours:
        (x, y, w, h) = cv2.boundingRect(countour)
        
        if cv2.contourArea(countour) < 1200:
            continue
        
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
 
        if not contour_valid:
           continue
       
        hull = cv2.convexHull(countour)
        
        cv2.drawContours(frame_crop, hull, -1, (0, 255, 0), 3)
        
        M = cv2.moments(countour)
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if y < (line[1][1] + offset) and y > (line[1][1] - offset):
            roi_y, roi_h, roi_x, roi_w = (
                rect[1] + y, 
                rect[1] + y + h,
                rect[0] + x,
                rect[0] + x + w
            )
            
            detection_crop = image[roi_y: roi_h, roi_x: roi_w]   

            cv2.imshow('detection crop', detection_crop)
            cv2.rectangle(image, (roi_x, roi_y), (roi_w, roi_h), (255, 0, 0), 2)
            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
            cv2.drawMarker(frame_crop, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,line_type=cv2.LINE_AA)
                 
    cv2.imshow('original', image)
        
    key = cv2.waitKey(30)
    
    if key == ord("e"):
        break

cv2.destroyAllWindows()
cap.release()
        

        
        
        