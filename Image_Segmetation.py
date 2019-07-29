import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

video_capture = cv2.VideoCapture(0)

lower = 115
upper = 235
distance = 0.0
face_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascade_Files/haarcascade_smile.xml')
sign_cascade = cv2.CascadeClassifier('Cascade_Files/sign_cascade.xml')
traffic_light_cascade = cv2.CascadeClassifier('Cascade_Files/traffic_light.xml')
hand_cascade = cv2.CascadeClassifier('Cascade_Files/hand_cascde.xml')
palm_cascade = cv2.CascadeClassifier('Cascade_Files/rpalm_cascade.xml')
yield_sign_cascade = cv2.CascadeClassifier('Cascade_Files/yield_sign_cascade.xml')
traffic_light_cascade = cv2.CascadeClassifier('Cascade_Files/traffic_light_cascade.xml')

def detect(gray, frame):
    signs = sign_cascade.detectMultiScale(gray, 1.3, 5)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, 1.3, 5)
    palm = palm_cascade.detectMultiScale(gray, 1.3, 5)
    yield_sign = yield_sign_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
    traffic_light = traffic_light_cascade.detectMultiScale(gray, 1.3, 7)
   
   


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.2, 12)
        for (ex, ey, ew, eh) in eyes:
            cv2.putText(roi_color, 'Eyes', (ex, ey), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.putText(roi_color, 'Smile', (sx, sy), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    for (x, y, w, h) in signs:
        cv2.putText(frame, 'Signs', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in traffic_lights:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
        cv2.putText(frame, 'Traffic_Light', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
  
    for (x, y, w, h) in hands:
        cv2.putText(frame, 'Hand', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in palm:
        cv2.putText(frame, 'Palm', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
  
    for (x, y, w, h) in yield_sign:
        cv2.putText(frame, 'Yield_Sign', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

    for (x, y, w, h) in traffic_light:
        cv2.putText(frame, 'Traffic_Light', (x, y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)


    return frame



while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = cv2.Canny(gray, lower, upper)

    hav = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h = hav[:, :, 0]
    s = hav[:, :, 1]
    v = hav[:, :, 2]  

    lower_skin = np.array([60, 40, 70])
    higher_skin = np.array([ 200, 240, 200])

    mask_rgb = cv2.inRange(frame, lower_skin, higher_skin)
    cv2.imshow('Canny Edge Detector', canvas)
    detected = detect(gray, frame)
    cv2.imshow('Video', detected)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


'''
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    thresh = 0.11*dst.max()
    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if (dst[j, i] > thresh):
                cv2.circle(frame, (i, j), 2, (0, 255, 0), 1)    

# Sign Detection
#                      
    def detect_sign(gray, frame):
    signs = sign_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    return frame

'''


