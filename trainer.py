import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create();
path='C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\FACE IDENTIFICATION\\USERNAME'

def getImagesWithId(path):
    imgpaths=[os.path.join(path,f) for f in os.listdir(path)]
    #The os.listdir(path) is listing all the directories which are the pictures and fetching all the directories of the pictures
    #The join function concatenates path with a slash and creating a list
    faces=[]
    IDs=[]
    for imgpath in imgpaths:   #First we have to open the image and convert it to numpy array
        faceImg=Image.open(imgpath).convert('L');
        facenp=np.array(faceImg,np.uint8)
        ID=int(os.path.split(imgpath)[-1].split('.')[1])#First the image should be split using path spliter and then using '.' spliter.[-1] means happens from backwards
        faces.append(facenp)
        IDs.append(ID)
        cv2.imshow("Training",facenp)
        cv2.waitKey(10)
    return np.array(IDs),faces

IDs,faces=getImagesWithId(path)
recognizer.train(faces,IDs)
recognizer.write('C:\\Users\\B.Vishnu charan\\Desktop\\VISHNU FILES\\FOURTH SEMESTER\\IMAGE PROCESSING\\FACE IDENTIFICATION\\TRAINER\\traindata.yml')
cv2.destroyAllWindows()