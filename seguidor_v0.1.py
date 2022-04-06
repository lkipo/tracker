import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame = cap.read()

#cv.imshow('Image', im)

r = cv.selectROI(im)

imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv.imshow('Crop', imCrop)
cv.waitKey(0)