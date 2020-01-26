import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

capture = cv.VideoCapture(r'/Users/amitgupta/Downloads/videoplayback (3).mp4')
while True:
    ret,frame=capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    pts = np.array([[(75,mask.shape[0]),(mask.shape[1]//2,mask.shape[0]//2),(mask.shape[1]-100,mask.shape[0])]])
    cv.fillPoly(mask,pts, 255)
    segment = cv.bitwise_and(gray, mask)
    imgCanny = cv.Canny(segment, 150, 200)
    x = 75
    z = mask.shape[1]-100
    while imgCanny[x,mask.shape[0]] != 255 and x < z:
        x +=1
    if x > (z - x):
        cv.putText(frame, 'turn left', (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0))
    elif x < (z - x):
        cv.putText(frame, 'turn right',(50,50), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0))
    cv.imshow('frame',frame) 
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break 

capture.release()
cv.destroyAllWindows()

