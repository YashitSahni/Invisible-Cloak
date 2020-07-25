import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('hsv', hsv)
        
        red = np.uint8([[[0,0,255]]])
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        
        l_red = np.array([0,100,100])
        h_red = np.array([10,255,255])
        
        #This both lower and higher red its value has been taken from documentation
        #As it is mentioned on opencv documentation
        
        mask = cv2.inRange(hsv, l_red, h_red)
        #cv2.imshow('mask', mask)
        
        #Showing red
        part1 = cv2.bitwise_and(back, back, mask=mask)
        
        mask = cv2.bitwise_not(mask)
        
        #Not showing red
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        
        total = part1 + part2
        
        t = cv2.morphologyEx(total, cv2.MORPH_OPEN, (10,10))
        
        dst = cv2.fastNlMeansDenoisingColored(t, None, 10, 10, 7, 15) 
        
        #cv2.imshow('cloak', t)
        cv2.imshow('real', dst)
        
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()            
cv2.destroyAllWindows()