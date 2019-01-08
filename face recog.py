import imutils
from imutils import face_utils
import dlib
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
if cap.isOpened():
    ret,frame=cap.read()
else:
    print("Unable to detect camera ")


fileloc='C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\FACE IDENTIFICATION\\EXPRESSION\\expr.jpg'
cv2.imwrite(fileloc,frame)
img1=imutils.resize(frame,width=500)
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\HAAR CASCADES\\shape_predictor_68_face_landmarks.dat")

# detect faces in the grayscale image
rects=detector(gray,1)

# loop over the face detections
for (i,rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape=predictor(gray,rect)
    shape=face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x,y,w,h)=face_utils.rect_to_bb(rect)
    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.putText(img1,'Face#{}'.format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x,y) in shape:
        cv2.circle(img1,(x,y),1,(0,0,255),-1)

cv2.imshow("Output",img1)
cv2.waitKey(0)

def rect_to__bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y

    return (x,y,w,h)

def shape_to_np(shape,dtype='int'):
    # initialize the list of (x, y)-coordinates
    coords=np.zeros((68,2),dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0,68):
        coords[i]=(shape.part(i).x,shape.part(i).y)

    return coords




