#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dlib
import imutils
import time
import cv2
import Hist_equalization as Heq
import faceangle

from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
from scipy.spatial import distance as dist


#EAR 정의
def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EAR_thresh = 0 #EAR 기준값
AV=0 #평균 EAR 축적
EAR_frames = 25 #EAR 측정 프레임 간격
count=0

Angle_Threshold=0 #ANGLE 기준값
anglesum=0 #angele 값 축적
avgangle=0 #angle 평균

EAR_ALARM_FLAG = False #EAR 알람 신호
ANGLE_ALARM_FLAG = False #ANGLE 알람 신호

print("detector start...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("video stream loading...")
vs = VideoStream(src=0).start()
time.sleep(1.0)


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 450) 
    gray = Heq.His_equalization(frame)
    rects = detector(gray,0) 

    if len(rects)==0 : # 감지된 얼굴 수
        cv2.putText(frame, "NOT detected", (40,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    else :
        cv2.putText(frame, "rects : {:.2f}".format(len(rects)), (40,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    for rect in rects:
        #1 눈감지
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average 
        both_ear = (leftEAR + rightEAR)/2 
        
        AV+=both_ear 
        
        if(count==25): #25번으로 평균 측정
            EAR_thresh = (AV/25)*0.8 #EAR THRESH 기준 설정
            AV = 0
            
        cv2.putText(frame, "EAR_Threshold : {:.2f}".format(EAR_thresh), (200,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
        
        count+=1
        #2 얼굴각도
        facialpoints = [shape[33], shape[8], shape[36], shape[45], shape[48], shape[54]]
        angle = faceangle.faceangle(gray, facialpoints)

        anglecount = 50
        
        if count % anglecount != (anglecount - 1):
            anglesum += angle 
        
        elif count == (anglecount - 1):
            avgangle = (anglesum/anglecount)
            Angle_Threshold=avgangle
            anglesum = 0

        else:
            avgangle = (anglesum/anglecount)
            anglesum = 0
        
        cv2.putText(frame, "Angle_Threshold : {:.2f}".format(Angle_Threshold), (200,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Angle : {:.2f}".format(avgangle), (300,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
        ###
        
        if both_ear < EAR_thresh :
            EAR_ALARM_FLAG += 1 
        else :           
            EAR_ALARM_FLAG = False
            
        if avgangle < Angle_Threshold -3.5:
            ANGLE_ALARM_FLAG =1
        else:
            ANGLE_ALARM_FLAG = False
            
            
        ### ALARM SYSTEM     
        if EAR_ALARM_FLAG >= EAR_frames and ANGLE_ALARM_FLAG == 0: 
            cv2.putText(frame, "****************ALARM 1****************", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
            cv2.putText(frame, "****************ALARM 1****************", (10,300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
                
        if EAR_ALARM_FLAG < EAR_frames and ANGLE_ALARM_FLAG == 1 :      
            cv2.putText(frame, "****************ALARM 2****************", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
            cv2.putText(frame, "****************ALARM 2****************", (10,300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        if EAR_ALARM_FLAG >= EAR_frames and ANGLE_ALARM_FLAG == 1 :      
            cv2.putText(frame, "****************DROWSY!****************", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            cv2.putText(frame, "****************DROWSY!****************", (10,300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
    #show the frame
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

#종료
cv2.destroyAllWindows()
vs.stop()

