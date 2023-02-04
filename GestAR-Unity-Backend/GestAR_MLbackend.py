import cv2
from cvzone.HandTrackingModule import HandDetector

#
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui

#


import numpy as np
import tensorflow as tf

#

path = cv2.CascadeClassifier("Face_Recognition/haar_cascade_face_detection.xml")

# Initiate video capture for video file


mouse = Controller()

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

(screen_width, screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#
mymodel = tf.keras.models.load_model("model3.h5")

detector = HandDetector(detectionCon=0.8, maxHands=2)
vals = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    " ",
]

#

handsms = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

#

tipIds = [4, 8, 12, 16, 20]

pinch = False

#


def countFingers(image, hand_landmarks, handNo=0):

    global pinch

    if hand_landmarks:
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[handNo].landmark

        # Count Fingers
        fingers = []

        for lm_index in tipIds:
            # Get Finger Tip and Bottom y Position Value
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y

            # Check if ANY FINGER is OPEN or CLOSED
            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)

                if finger_tip_y > finger_bottom_y:
                    fingers.append(0)

        totalFingers = fingers.count(1)

        # PINCH

        # Draw a LINE between FINGER TIP and THUMB TIP
        finger_tip_x = int((landmarks[8].x) * width)
        finger_tip_y = int((landmarks[8].y) * height)

        thumb_tip_x = int((landmarks[4].x) * width)
        thumb_tip_y = int((landmarks[4].y) * height)

        cv2.line(
            image,
            (finger_tip_x, finger_tip_y),
            (thumb_tip_x, thumb_tip_y),
            (255, 0, 0),
            2,
        )

        # Draw a CIRCLE on CENTER of the LINE between FINGER TIP and THUMB TIP
        center_x = int((finger_tip_x + thumb_tip_x) / 2)
        center_y = int((finger_tip_y + thumb_tip_y) / 2)

        cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 2)

        # Calculate DISTANCE between FINGER TIP and THUMB TIP
        distance = math.sqrt(
            ((finger_tip_x - thumb_tip_x) ** 2) + ((finger_tip_y - thumb_tip_y) ** 2)
        )

        # print("Distance: ", distance)


        # █▀ █▀▀ █▀█ █▀▀ █▀▀ █▄░█   █▀ █ ▀█ █▀▀
        # ▄█ █▄▄ █▀▄ ██▄ ██▄ █░▀█   ▄█ █ █▄ ██▄
        # print(
        #     "Computer Screen Size :",
        #     screen_width,
        #     screen_height,
        #     "Output Window size: ",
        #     width,
        #     height,
        # )


        # █▀▄▀█ █▀█ █░█ █▀ █▀▀   █▀█ █▀█ █▀
        # █░▀░█ █▄█ █▄█ ▄█ ██▄   █▀▀ █▄█ ▄█
        # print(
        #     "Mouse Position: ",
        #     mouse.position,
        #     "Tips Line Centre Position: ",
        #     center_x,
        #     center_y,
        # )

        # Set Mouse Position on the Screen Relative to the Output Window Size
        relative_mouse_x = (center_x / width) * screen_width
        relative_mouse_y = (center_y / height) * screen_height

        mouse.position = (relative_mouse_x, relative_mouse_y)


        # █▀█ █ █▄░█ █▀▀ █░█ █▀▀ █▀▄ ▀█
        # █▀▀ █ █░▀█ █▄▄ █▀█ ██▄ █▄▀ ░▄

        if distance > 40:
            if pinch == True:
                pinch = False
                # mouse.release(Button.left)

        if distance <= 40:
            if pinch == False:
                pinch = True
                # mouse.press(Button.left)


def max_and_index(arr):
    m = arr[0]
    ind = 0
    for index, i in enumerate(arr):
        if i > m:
            m = i
            ind = index
    return (vals[ind], m)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = path.detectMultiScale(gray, 1.1, 1)
    results = handsms.process(img)
    hands, img = detector.findHands(img)  # With Draw
    #
    #
    # Get landmark position from the processed result
    hand_landmarks = results.multi_hand_landmarks


    # █▀▀ ▄▀█ █▀▀ █▀▀   █▄▄ █▀█ █░█ █▄░█ █▀▄ █ █▄░█ █▀▀   █▄▄ █▀█ ▀▄▀
    # █▀░ █▀█ █▄▄ ██▄   █▄█ █▄█ █▄█ █░▀█ █▄▀ █ █░▀█ █▄█   █▄█ █▄█ █░█
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Get Hand Fingers Position
    countFingers(img, hand_landmarks)
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # Hand Type Left or Right

        # print(len(lmList1),lmList1)
        # print(bbox1)
        # print(centerPoint1)
        fingers1 = detector.fingersUp(hand1)
        # length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) # with draw
        # length, info = detector.findDistance(lmList1[8], lmList1[12])  # no draw

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left or Right

            fingers2 = detector.fingersUp(hand2)
            
            # █▀▀ █ █▄░█ █▀▀ █▀▀ █▀█ █▀
            # █▀░ █ █░▀█ █▄█ ██▄ █▀▄ ▄█
            print(fingers1, fingers2)
            # format of the array:
            # [[left hand],[right hand]]
            # [[thumb, index, middle, ring, pinky],[thumb, index, middle, ring, pinky]]
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) # with draw
            length, info, img = detector.findDistance(
                centerPoint1, centerPoint2, img
            )  # with draw

            # █▀▄ █ █▀ ▀█▀ ░   █▄▄ █▀▀ ▀█▀ █░█░█ █▀▀ █▀▀ █▄░█   █▀▀ █ █▄░█ █▀▀ █▀▀ █▀█ █▀
            # █▄▀ █ ▄█ ░█░ ▄   █▄█ ██▄ ░█░ ▀▄▀▄▀ ██▄ ██▄ █░▀█   █▀░ █ █░▀█ █▄█ ██▄ █▀▄ ▄█
            # print(length)


        if success:

            # Flip the img

            # Resize the img
            resized_img = cv2.resize(img, (224, 224))

            # Expanding the dimension of the array along axis 0
            resized_img = np.expand_dims(resized_img, axis=0)

            # Normalizing for easy processing
            resized_img = resized_img / 255

            # Getting predictions from the model
            predictions = mymodel.predict(resized_img)

            # printing percentage confidence

            
            # █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █▀▀ █▀▄   ▄▀█ █▀ █░░   █▀▀ █ █▄░█ █▀▀ █▀▀ █▀█
            # █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ ██▄ █▄▀   █▀█ ▄█ █▄▄   █▀░ █ █░▀█ █▄█ ██▄ █▀▄
            # print(max_and_index(predictions[0]))

    cv2.imshow("output", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
