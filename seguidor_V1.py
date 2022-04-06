import cv2 as cv
import numpy as np

camshift = False
pause = False
crop = False

def init_camshift(x, y, w, h):
    # Probar esto ↓
    track_window = (x, y, w, h)

    ret, frame = cap.read()

    # Estas 5 lineas estan copiadas xd
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    return roi, hsv_roi, mask, roi_hist, term_crit, track_window

# cap = cv.VideoCapture('helicopter.mp4') # capturar video
cap = cv.VideoCapture(0) # capturar camara 0

while True:
    # Lemos a info da camara
    ret, frame = cap.read()

    # Control ↓
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
        
    if crop:
        r = cv.selectROI(frame, fromCenter=False)
        camshift = True
        crop = False
        cv.destroyWindow('Frame')
        cv.destroyWindow('ROI selector')

        roi, hsv_roi, mask, roi_hist, term_crit, track_window = init_camshift(int(r[0]), int(r[1]), int(r[2]), int( r[3]))

    if camshift:
        print(track_window)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1) # nin idea de que fai esto
        
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Mostramos a imaxe
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        #cv.imshow('hsv', hsv)

    else:
        cv.imshow('frame', frame)