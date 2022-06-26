import numpy as np
import cv2

image = cv2.imread('./images/image.jpg')
clone = image.copy()
points = []
count = 0

def crop_points(img):
    mask = np.zeros(img.shape[0:2], dtype = np.uint8)
    
    points = np.array([points])
    
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_8)
    
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    
    cv2.imshow("crop", cropped)
# left click to draw the next line, right click to delete the previous line
def draw_line(event, x, y, flags, param):
    global points, count, image
    
    if event == cv2.EVENT_LBUTTONDOWN:  
        if count == 5:
            return
              
        points.append((x, y))
        
        cv2.circle(image, points[count - 1], 2, (255,0,0), 2)
        cv2.line(image, points[count - 1], points[count], (0, 255, 0), 2)
        
        if count == 3:
            first_pt = points[0]
            last_pt = points[count]
            
            cv2.circle(image, last_pt, 2, (255,0,0), 2)
            cv2.line(image, first_pt, last_pt, (0, 255, 0), 2)

        count = count + 1

    elif event == cv2.EVENT_RBUTTONDOWN:      
        if len(points) == 0:
            return
        
        image = clone.copy()   
         
        points.remove(points[count-1])
        
        count = count - 1
        
        for i in range (1, count):
            cv2.circle(image, points[i - 1], 2, (255,0,0), 2)
            cv2.line(image, points[i - 1], points[i], (0,255,0), 2)
            
        if len(points) == 1:
            count = 0
            points.clear()

cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_line)

while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'reset' key is pressed, reset to original state
    if key == ord("s"):
        crop_points(clone.copy())
        
    if key == ord("r"):
        image = clone.copy()    
        points = []
        count = 0
    # if the 'exit' key is pressed, break from the loop
    elif key == ord("e"):
        break

cv2.destroyAllWindows() 