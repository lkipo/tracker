import numpy as np
import cv2 as cv
import glob

    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

# image name
imname = 'calib_radial.jpg'
square_size = 10 # cm

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = square_size * np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv.imread(imname)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (7,6), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (7,6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread(imname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv.imwrite('calibresult.png', dst)

    cv.imshow('dist', dst)
    cv.waitKey()

    print(newcameramtx)

    # Estaría ben saber usar esto -> np.savez(file='B',  dst='dst', dist='dist', rvecs='rvecs', tvecs='tvecs')
 
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
    cv.imshow('img',img)
    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(imname[:6]+'.png', img)


cv.destroyAllWindows()