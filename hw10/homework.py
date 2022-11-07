#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
from matplotlib import pyplot as plt

# Load the dataset
folder = 'data/frames'
video = cv2.VideoCapture('data/video.mp4')


i = 0;
while True:
    ret, frame = video.read()
    if (i % 15) == 0:
        cv2.imwrite('data/frames/' + str(i) + '.jpg', frame)
    elif not ret:
        break
    i += 1

frames = os.listdir(folder)
frames.sort(key=lambda f: int(''. join(filter(str. isdigit, f))))
idx = frames.index(frames[0])


# Let's assume the detector has detected a vehicle
x1, y1 = 205, 335
x2, y2 = 345, 595


width = x2 - x1
height = y2 - y1

# Set up tracker
tracker_types = ['MIL','KCF', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()

if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()

if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Genrate tracking template
img = cv2.imread(os.path.join(folder, frames[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize tracker
bbox = (x1, y1, width, height)
ok = tracker.init(img, bbox)


# Tracking loop
for ii in range(len(frames)):
    img = cv2.imread(os.path.join(folder, frames[ii]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        
    ok, bbox = tracker.update(img)
    print(ok, bbox)
               
    x1, y1 = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show(), plt.draw()    
    plt.waitforbuttonpress(0.1)
    plt.clf()
    
        
    