import cv2 as cv
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def smooth(s,x):
    return gaussian_filter(x,s,mode='constant')

REDU = 8

def rgbh(xs,mask):
    def normhist(x):
        return x / np.sum(x)

    def h(rgb):
        return cv.calcHist([rgb]
                            , [0, 1, 2]
                            , imCropMask
                            , [256//REDU, 256//REDU, 256//REDU]
                            , [0, 256] + [0, 256] + [0, 256]
                            )
    return normhist(sum(map(h, xs)))

cap = cv.VideoCapture(0) # 0 para capturar en directo, ruta do arquivo en outros casos

kernel = np.ones((3, 3), np.uint8)
bgsub = cv.createBackgroundSubtractorMOG2(500, 60, True)

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

pause= False
crop = False
camshift = False

while(True):
    key = cv.waitKey(30) & 0xFF
    if key== ord("c"): 
        crop = True
    if key== ord("p"): 
        P = np.diag([100,100,100,100])**2
    if key== ord("s"): 
        break
    if key==ord(" "): 
        pause =not pause
    if(pause): 
        continue


    ret, frame = cap.read()
    #frame=cv.resize(frame,(800,600))
    frame=cv.resize(frame,(1366,768)) # axustar para o tamaño da cámara ou video

    bgs = bgsub.apply(frame)
    bgs = cv.erode(bgs,kernel,iterations = 1)
    bgs = cv.medianBlur(bgs,3)
    bgs = cv.dilate(bgs,kernel,iterations=2)
    
    bgs = (bgs > 200).astype(np.uint8)*255
    colorMask = cv.bitwise_and(frame,frame,mask = bgs)
    
    if crop:
        fromCenter= False
        img = frame # Imaxe sobre a que seleccionar

        r = cv.selectROI(img, fromCenter)
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        crop = False
        camshift = True

        imCropMask = cv.cvtColor(imCrop, cv.COLOR_BGR2GRAY)
        ret,imCropMask = cv.threshold(imCropMask,30,255,cv.THRESH_BINARY)
        his = smooth(1,rgbh([imCrop],imCropMask))
        roiBox = (int(r[0]), int(r[1]),int(r[2]), int(r[3]))

        cv.destroyWindow("ROI selector")

    if camshift:

        rgbr = np.floor_divide(colorMask, REDU)
        r, g, b = rgbr.transpose(2, 0, 1)
        
        l = his[r, g, b]
        maxl = l.max()
        
        (rb, roiBox) = cv.CamShift(1, roiBox, term_crit)
        cv.rectangle(frame, (roiBox[0], roiBox[1]), (roiBox[2], roiBox[3]), (0, 0, 255), 2)
        print(roiBox)

    cv.imshow('Color Mask', colorMask)

    cv.imshow('mask', bgs)

    cv.imshow('Frame', frame)