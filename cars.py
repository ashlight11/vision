import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect cars
    cars = cars_cascade.detectMultiScale(frame_gray, scaleFactor=1.10, minNeighbors=3)
    for (x,y,w,h) in cars:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('Capture - Cars detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--cars_cascade', help='Path to face cascade.', default='./XML_models/cars.xml')
args = parser.parse_args()

cars_cascade_name = args.cars_cascade

cars_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not cars_cascade.load(cv.samples.findFile(cars_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)


#-- 2. Read the video stream
cap = cv.VideoCapture('./data/dataset_video1.avi')
 
while(cap.isOpened()):
    ret, frame = cap.read()
 
    if(ret):
        detectAndDisplay(frame)
        #cv.imshow('frame', frame)
        cv.waitKey()
 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 
 
cap.release()
cv.destroyAllWindows()