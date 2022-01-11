from __future__ import print_function
import cv2 as cv
import argparse
import numpy

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    color_info = (255, 255, 255)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, minNeighbors = 25, scaleFactor = 1.2)
    for (x,y,w,h) in faces:
        crop_frame = frame[y:y+h, x:x+w]
        median = numpy.median(crop_frame)
        if median > 135 : 
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # process the detections
            #print(str(x) +" " + str(y) + " " +str(w) + " " +str(h))
            cv.putText(crop_frame,"{:f}".format(median), (10, 30), cv.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv.LINE_AA)

            #cv.imshow("cropped", crop_frame)


            #cv.waitKey(1500)
        
        
    cv.imshow('Capture - Cola detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='./XML_models/cascade.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break