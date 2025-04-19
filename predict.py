from keras import models
import cv2 as cv
import numpy as np
import threading
import winsound
from pygame import mixer
# import os

print("\n---------------------Loading model---------------------\n")
model = models.load_model('crime-detection-model-2')
print("\n---------------------Model loaded---------------------\n")


def play_alarm_sound_function():
    mixer.music.play(loops=-1)


img_height, img_width = 180, 180
og_labels = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
             'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents',
             'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
# suspicious and normal actvity messages
sus = "!Suspicious_activity!"
warn = "*****WARNING*****"
norm = "Normal_activity"
buffer = []
# alarm properties
Alarm_Status = False
mixer.init()
mixer.music.load('alarm-sound.mp3')
T = threading.Thread(target=play_alarm_sound_function)
# font
font = cv.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# green color in BGR
green = (0, 255, 0)
# red color in BGR
red = (0, 0, 255)
# Line thickness of 2 px
thickness = 2
# execution_path = os.getcwd()

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('V_100.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    resize = cv.resize(frame, (img_height, img_width))
    image = np.expand_dims(resize, axis=0)
    pred = model.predict(image)
    # print(pred)
    frame_shape = np.shape(frame)
    # print(frame_shape)
    output_class = og_labels[np.argmax(pred)]
    # print("The predicted class is", output_class)
    if output_class != "NormalVideos":
        print("The predicted class is", sus, warn)
        # Using cv2.putText() method
        if len(buffer) < 10:
            buffer.append(sus)
            winsound.Beep(1500, 500)
        frame = cv.putText(frame, sus, org, font,
                           fontScale, red, thickness, cv.LINE_AA)
    else:
        print("The predicted class is", norm)
        # Using cv2.putText() method
        if len(buffer) != 0:
            buffer.pop()
        frame = cv.putText(frame, norm, org, font,
                           fontScale, green, thickness, cv.LINE_AA)

    if frame_shape[1] >= 960 and frame_shape[0] >= 540:
        frame = cv.resize(frame, (frame_shape[1]//2, frame_shape[0]//2))

    # cv.imwrite("result_image.png",frame)
    # Display the resulting frame
    cv.imshow('Video capture', frame)
    if len(buffer) == 10:
        print("Playing alarm")
        if Alarm_Status == False:
            T.run()
            Alarm_Status = True
            warn = "*****ALERT*****"
    if cv.waitKey(1) == ord('q'):
        mixer.music.stop()
        break
# When everything done, release the capture
mixer.quit()
cap.release()
cv.destroyAllWindows()
