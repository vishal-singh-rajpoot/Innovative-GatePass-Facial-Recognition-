import numpy as np
import sqlite3
import cv2,os
import pickle
from PIL import Image


recognizer=cv2.face_LBPHFaceRecognizer.create();
recognizer.read("trainer/training_data.yml")
cascadePath="Classifiers/face.xml"
faceCascade=cv2.CascadeClassifier(cascadePath);
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path='dataSet'

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People where ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

    
cam=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        

        profile=getProfile(id)

        if conf<60:
            if(profile!=None):    
                cv2.putText(im, "Name: "+str(profile[1]), (x, y+h+30), font, 0.4, (0, 0, 255), 1);
                cv2.putText(im, "Age: " + str(profile[2]), (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
                cv2.putText(im, "Gender: " + str(profile[3]), (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
                cv2.putText(im, "Criminal Records: " +str(profile[4]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
                
        
        else:
            cv2.putText(im, "Name: Unknown", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
            cv2.putText(im, "Age: Unknown", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
            cv2.putText(im, "Gender: Unknown", (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
            cv2.putText(im, "Criminal Records: Unknown", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
        
        
    cv2.imshow('im',im)
    
    cv2.waitKey(10)
        
    
