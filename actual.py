import cv2 as cv
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

with open(r'final_modelrf.pkl', 'rb') as f:
    model = pickle.load(f)
mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

cam=cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret,frame=cam.read()
        #Change color so that mediapipe can process it
        rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        #mediapipe processing it
        processing_result=hol.process(rgb_frame)
        #making color again0
        color_frame=cv.cvtColor(rgb_frame,cv.COLOR_RGB2BGR)
        #  print(processing_result.face_landmarks)
        #Drawing the landmarks of Face
        mp_drawing.draw_landmarks(color_frame,processing_result.face_landmarks,mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(255,0,0),thickness=1,circle_radius=2),mp_drawing.DrawingSpec(color=(0,0,0),thickness=1,circle_radius=1))

        #Drawing the landmarks of Pose
        mp_drawing.draw_landmarks(color_frame,processing_result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        #Drawing the landmarks of Left hand
        #mp_drawing.draw_landmarks(color_frame,processing_result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2))

        #Drawing the landmarks of Right hand
        #mp_drawing.draw_landmarks(color_frame,processing_result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        
        # Extract Pose landmarks
        if(processing_result.pose_landmarks==None) and(processing_result.face_landmarks==None):
            continue
        pose = processing_result.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Extract Face landmarks
        face = processing_result.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        # Concate rows
        row = pose_row+face_row
        
        a=pd.DataFrame([row])
        body_language_class = model.predict(a)[0]
        body_language_prob = model.predict_proba(a)[0]
        print(body_language_class, body_language_prob)
        coords = tuple(np.multiply(
                            np.array(
                                (processing_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                processing_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
        cv.rectangle(color_frame, 
                        (coords[0], coords[1]+5), 
                        (coords[0]+len(body_language_class)*20, coords[1]-30), 
                        (245, 117, 16), -1)
        cv.putText(color_frame, body_language_class, coords, 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
            # Get status box
        cv.rectangle(color_frame, (0,0), (250, 60), (255, 255,0), -1)
            
            # Display Class
        cv.putText(color_frame, 'CLASS'
                        , (95,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(color_frame, body_language_class.split(' ')[0]
                        , (90,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
            # Display Probability
        cv.putText(color_frame, 'PROB'
                        , (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(color_frame, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        
        cv.imshow('camera',color_frame)
        if cv.waitKey(100) & 0xFF==ord('a') :
            break
        

cam.release
cv.destroyAllWindows()