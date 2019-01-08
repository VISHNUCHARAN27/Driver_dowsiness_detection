import cv2
import numpy as np


face_cascade=cv2.CascadeClassifier("C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\ALA\\PPTS\\VIDEO TRY\\haarcascade_frontalface_default.xml")
img=cv2.imread("C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\DATASETS\\4.1.04.tiff")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.02,minNeighbors=5)
#detectMultiScale() is a method to search for the face rectangle co-ordinates
#scaleFactor decreases the shape value by 5% until the face is found. Smaller the value greater is the accuracy
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    #Method to create a face rectangle
    #(x,y),(x+w,y+h) are the co-ordinates
    #(0,255,0) is the color and 3 is the thickness
resized=cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))
cv2.imshow("Gray",resized)
cv2.waitKey(0)

