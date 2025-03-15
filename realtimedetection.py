import cv2
import cv2.data
from keras.models import model_from_json
import numpy as np
import os
import time
import sys
# from keras.preprocessing.image import load_img
json_file = open('emotiondetector.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('emotiondetector.h5')
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
while True:
   i,im  = webcam.read()
   gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(im, 1.3, 5)
   try:
      for (x, y, w, h) in faces:
          image = gray[y:y+h, x:x+w]
          cv2.rectangle(im,(x, y),(x+w, y+h),(255, 0, 0), 2)
          image = cv2.resize(image, (48, 48))
          img = extract_features(image)
          pred = model.predict(img)
          prediction_label = labels[pred.argmax()]
          cv2.putText(im, prediction_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
      cv2.imshow("Output", im)
      cv2.waitKey(27)
   except cv2.error:
      pass