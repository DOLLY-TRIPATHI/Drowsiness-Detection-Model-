import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame, threading, time

# Load trained model
model = load_model("drowsiness_model.h5")

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
score = 0
last_alert_time = 0  

# Sound alert
def play_alert():
    global last_alert_time
    if time.time() - last_alert_time > 5:  
        last_alert_time = time.time()
        pygame.mixer.init()
        pygame.mixer.music.load(r"D:/Driver_Drowsiness_Project/alert.mp3")
        pygame.mixer.music.play()

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face)

        # sirf 2 sabse bade eyes rakho
        eyes = sorted(eyes, key=lambda ex: ex[2]*ex[3], reverse=True)[:2]

        preds = []
        for (ex,ey,ew,eh) in eyes:
            eye = roi_face[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24,24))
            eye = eye.astype("float")/255.0
            eye = img_to_array(eye)
            eye = np.expand_dims(eye, axis=0)

            pred = model.predict(eye, verbose=0)[0][0]
            preds.append(pred)

        # Ek hi decision frame ke liye
        if len(preds) > 0:
            avg_pred = np.mean(preds)
            if avg_pred > 0.5:   # Open
                score = max(0, score-1)
                cv2.putText(frame, 'Open', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
            else:               # Closed
                score += 1
                cv2.putText(frame, 'Closed', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            if score > 15:
                cv2.putText(frame, "DROWSINESS ALERT!", (100,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
                threading.Thread(target=play_alert, daemon=True).start()

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
