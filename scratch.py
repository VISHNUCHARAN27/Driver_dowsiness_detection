import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\HAAR CASCADES\\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\HAAR CASCADES\\haarcascade_eye.xml")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('Image',frame)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()

