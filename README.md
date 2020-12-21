# AI-base-self-driving-car-part-01
This is image detection part of real time video. After we can train neural network . so this codes are About vehicle, pedestrian &amp; sign board detection. this is a base of computer visson

Computer vision is a field of study which encompasses on how computer see and understand digital images and videos.
Computer vision involves seeing or sensing a visual stimulus, make sense of what it has seen and also extract complex information that could be used for other machine learning activities.
This is one of the most important applications of Computer vision where the self-driving cars need to gather information about their surroundings to decide how to behave.try to understand what these Haar Cascade Classifiers are.This is basically a machine learning based approach where a cascade function is trained from a lot of images both positive and negative. Based on the training it is then used to detect the objects in the other images.
So how this works is they are huge individual .xml files with a lot of feature sets and each xml corresponds to a very specific type of use case.

Vehicle detection from streaming video
Let’s implement one more use case from the haar cascade classifier. In this use-case we will be detecting the vehicles from a streaming video.I have implemented these use-cases to show how it works. There are whole bunch of other xmls also for this classifier which you could use to implement to implement few other computer vision cases as well. Here is the github link for the xmls.
The implementation here is same as the one we did for face detection, so I won’t be going in detail explaining the whole process. However there are couple of changes that are there in the code.
Step 1
In order to detect the features of a vehicle we need to import the haarcascade_car.xml.
Use the VideoCapture of cv2 and store the value in cap
Reading (cap.read()) from a VideoCapture returns a tuple (ret, frame). With the first item you check whether the reading was successful, and if it was then you proceed to use the returned frame.
cap = cv2.VideoCapture('/vehicle.mp4')
# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
Step 2
Now that we have the tuple of (ret, frame) , we will convert the BGR channel image to gray channel. Reasons being the same, we are converting the image to gray scale and using the classifier function detectMultiScale to extract the x-coordinate, y-coordinate, width (w) and height(h), and gray scale is used for better performance throughput.
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.1, 2)
Step 3
Based on the extracted features/dimensions of the cars, we will loop through them and draw a rectangle around each frame of the image.
for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
Below is the consolidated code for that:
import time
import numpy as np
import cv2
# Create our body classifier
car_classifier = cv2.CascadeClassifier('\haarcascade_car.xml')
# Initiate video capture for video file
cap = cv2.VideoCapture('/vehicle.mp4')
# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.1, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)
if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
Now that we have the complete code, let’s check the output for it.

*** Pedestrian detection from a streaming video
The implementation is completely the same as the vehicle detection. The only difference here will be, we will be using the haarcascade_fullbody.xml to identify the features of the pedestrian’s body.
Below is the code for it:
import numpy as np
import cv2
# Create our body classifier
body_classifier = cv2.CascadeClassifier('\haarcascade_fullbody.xml')
# Initiate video capture for video file
cap = cv2.VideoCapture('/moskva.mov')
# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)
if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
