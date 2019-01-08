import imutils
import cv2
import os
imgpath='C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\DATASETS\\lena_color_512.tif'
img=cv2.imread(imgpath,1)
cv2.circle(img,(90,90),40,(255,0,0),2)
cv2.imshow('Lena',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

