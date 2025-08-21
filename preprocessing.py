import cv2 as cv
import mediapipe as mp
import csv
import os
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

cam=cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    while True:
        ret,frame=cam.read()
        #Change color so that mediapipe can process it
        rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        #mediapipe processing it
        processing_result=hol.process(rgb_frame)
        #making color again
        color_frame=cv.cvtColor(rgb_frame,cv.COLOR_RGB2BGR)
        #  print(processing_result.face_landmarks)
        #Drawing the landmarks of Face
        mp_drawing.draw_landmarks(color_frame,processing_result.face_landmarks,mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(255,0,0),thickness=1,circle_radius=2),mp_drawing.DrawingSpec(color=(0,0,0),thickness=1,circle_radius=1))

        #Drawing the landmarks of Pose
        mp_drawing.draw_landmarks(color_frame,processing_result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        #Drawing the landmarks of Left hand
        mp_drawing.draw_landmarks(color_frame,processing_result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2))

        #Drawing the landmarks of Right hand
        mp_drawing.draw_landmarks(color_frame,processing_result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        if(processing_result.pose_landmarks==None):
            continue
        num_coords = len(processing_result.pose_landmarks.landmark)

        print(num_coords)

        landmarks = ['class']
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        with open('pose.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

        class_name = "victory"
        # Extract Pose landmarks
        try:
            pose = processing_result.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face landmarks
            face = processing_result.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        # Concate rows
            row = pose_row+face_row
        
        
        # Append class name 
            row.insert(0, class_name)
            #print(row)
        
        # Export to CSV
            with open('v.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
        except:
            pass
        cv.imshow('camera',color_frame)
        if cv.waitKey(100) & 0xFF==ord('a') :
            break
        

cam.release
cv.destroyAllWindows()