import cv2
from cv2 import bitwise_and
import numpy as np

cap = cv2.VideoCapture(0)

kernel = np.ones(5)
while True:
    _, frame = cap.read()
    img  = frame
    # Threshold
    grises = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:,:,1] = 255
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    red = frame[:,:,2]
    blue = frame[:,:,1]
    green = frame[:,:,0]
    _, tred = cv2.threshold(red, 150, 255, cv2.THRESH_BINARY)
    _, tgreen = cv2.threshold(green, 150, 255, cv2.THRESH_BINARY)
    _, tblue = cv2.threshold(blue, 150, 255, cv2.THRESH_BINARY)
    _, tgrises = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY_INV)

    filter = bitwise_and(tred, tgrises)
    # _, threshold = cv2.threshold(filter, 150, 255, cv2.THRESH_BINARY)

    filter = cv2.erode(filter, kernel)
    
    cv2.imshow("Frame", blue)
    cv2.imshow('rojo', tred)
    cv2.imshow("filter", filter)
    #cv2.imshow('tblue', tblue)
    #cv2.imshow('tgreen', tgreen)
    cv2.imshow('grisses', tgrises)
    
    contours, _= cv2.findContours(filter, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Calculate area
        area = cv2.contourArea(cnt)
        
        # Distinguish small and big nuts
        if area > 400:
            # big nut
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    key = cv2.waitKey(1)
    if key == 115:
        break
    cv2.imshow('img', img)
cap.release()
cv2.destroyAllWindows()