import cv2
from cvzone.HandTrackingModule import HandDetector
#
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui
import os
#
import numpy as np
import tensorflow as tf
#
import json

path = cv2.CascadeClassifier("haar_cascade_face_detection.xml")

# Initiate video capture for video file


mouse = Controller()

cap = cv2.VideoCapture(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

(screen_width, screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands

print("type hands", type(mp_hands), mp_hands)

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

        # Peace

        # Draw a LINE between FINGER TIP and THUMB TIP
        finger_tip_x = int((landmarks[8].x) * width)
        finger_tip_y = int((landmarks[8].y) * height)

        thumb_tip_x = int((landmarks[12].x) * width)
        thumb_tip_y = int((landmarks[12].y) * height)

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

        # mouse.position = (relative_mouse_x, relative_mouse_y)

        # █▀█ █ █▄░█ █▀▀ █░█ █▀▀ █▀▄ ▀█
        # █▀▀ █ █░▀█ █▄▄ █▀█ ██▄ █▄▀ ░▄
        # print("Dist: ", distance)
        # if distance <= 40:
        #     if pinch == True:
        #         pinch = False
        #         # mouse.release(Button.left)
        #     print("Left side: ", relative_mouse_y)

        # if distance <= 40:
        #     if pinch == False:
        #         pinch = True
        #         # mouse.press(Button.left)
        #     print("Right side: ", relative_mouse_y)

        print("line 180 FINGERS", fingers)


def getLandMarks(image, hand_landmarks, handNo=0):

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
        print("FINGERS", fingers, "\nHAND NO", handNo)
        return totalFingers


def max_and_index(arr):
    m = arr[0]
    ind = 0
    for index, i in enumerate(arr):
        if i > m:
            m = i
            ind = index
    return (vals[ind], m)


# ░░█ █▀ █▀█ █▄░█   █▀▀ █▀█ █▀█   █░█ █▄░█ █ ▀█▀ █▄█
# █▄█ ▄█ █▄█ █░▀█   █▀░ █▄█ █▀▄   █▄█ █░▀█ █ ░█░ ░█░

'''
WHAT TO INCLUDE
int[] CVRes; <-- screen resolution
ALL THE "box" ONES HAVE CENTER X, CENTER Y, WIDTH, HEIGHT
Vector2[] headBox; <-- bounding box pos of head
Vector2[] hand1Box; <-- bounding box pos of hand 1
Vector2[] hand2Box; <-- bounding box pos of hand 2
bool rightAnd1Hand; <-- to tell if it's the right or left hand when only 1 hand is visible
bool[] hand1Fingers; <-- condition of hand 1 fingers closed or open
bool[] hand2Fingers; <-- condition of hand 2 fingers closed or open
string handASLText; <-- text generated from ASL model
int slider1Amt; <-- amount of slider 1 (0 to 10 inc)
int slider2Amt; <-- amount of slider 2 (0 to 10 inc)
'''

# open the file in the beginning so we can easily write to it continuously
jsonForUnity = open("jsonForUnity.json", "w")
dataForUnity = {
    "CVRes": [0, 0],
    "headBox": [[0, 0], [0, 0]],
    "hand1Box": [[0, 0], [0, 0]],
    "hand2Box": [[0, 0], [0, 0]],
    "hand1Fingers": [False, False, False, False, False],
    "hand2Fingers": [False, False, False, False, False],
    "handASLText": "...",
    "slider1Amt": 0,
    "slider2Amt": 0
}
def saveJsonForUnity():
    print("saving to json")
    j = json.dumps(dataForUnity)
    jsonForUnity.write(j)
def setJsonData(obj):
    for key in obj:
        val = obj[key]
        dataForUnity[key] = val;


# ████████╗██╗░░██╗███████╗
# ╚══██╔══╝██║░░██║██╔════╝
# ░░░██║░░░███████║█████╗░░
# ░░░██║░░░██╔══██║██╔══╝░░
# ░░░██║░░░██║░░██║███████╗
# ░░░╚═╝░░░╚═╝░░╚═╝╚══════╝
#     W   H   I   L   E
#       L   O   O   P

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = path.detectMultiScale(gray, 1.1, 3)
    results = handsms.process(img)
    hands, img = detector.findHands(img)  # With Draw

    print("INITIAL RES")
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print("FRAME WIDTH", frame_width)
    print("FRAME HEIGHT", frame_height)

    # Get landmark position from the processed result
    hand_landmarks = results.multi_hand_landmarks

    # █▀▀ ▄▀█ █▀▀ █▀▀   █▄▄ █▀█ █░█ █▄░█ █▀▄ █ █▄░█ █▀▀   █▄▄ █▀█ ▀▄▀
    # █▀░ █▀█ █▄▄ ██▄   █▄█ █▄█ █▄█ █░▀█ █▄▀ █ █░▀█ █▄█   █▄█ █▄█ █░█
    largestFace = 0
    i = 0
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # json data will get the face with largest area
        if w * h > face[0][2] * face[0][3]:
            largestFace = i
            setJsonData({"headBox": [[int(x) + int(w/2), int(y) + int(w/2)], [int(w), int(h)]]})
        i += 1

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
        if 0 < centerPoint1[0] < 180:
            print("L", (10-int(centerPoint1[1] / 40)))
            # 400(0) - 100(10)
        if 0 < centerPoint1[0] > 420:
            print("R", (10-int(centerPoint1[1] / 40)))

        fingers1 = detector.fingersUp(hand1)
        # length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) # with draw
        # length, info = detector.findDistance(lmList1[8], lmList1[12])  # no draw

        if len(hands) == 2:
            print("LANDMARKS", getLandMarks(img, hand_landmarks))

            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left or Right

            x=y=w=h=x2=y2=w2=h2=0
            if (handType1 == "Right"):
                x = int(bbox1[0])
                y = int(bbox1[1])
                w = int(bbox1[2])
                h = int(bbox1[3])

                x2 = int(bbox2[0])
                y2 = int(bbox2[1])
                w2 = int(bbox2[2])
                h2 = int(bbox2[3])
            else:
                x = int(bbox2[0])
                y = int(bbox2[1])
                w = int(bbox2[2])
                h = int(bbox2[3])

                x2 = int(bbox1[0])
                y2 = int(bbox1[1])
                w2 = int(bbox1[2])
                h2 = int(bbox1[3])

            setJsonData({"hand1Box": [[x + int(w/2), y + int(h/2)], [w, h]]})
            setJsonData({"hand2Box": [[x2 + int(w2/2), y2 + int(h2/2)], [w2, h2]]})

            fingers2 = detector.fingersUp(hand2)
            if 0 < centerPoint2[0] < 180:
                print("L", (10-int(centerPoint2[1] / 40)))
                # 400(0) - 100(10)
                # print("")
            if 0 < centerPoint1[0] > 420:
                # print("")
                print("R", (10-int(centerPoint1[1] / 40)))

            # █▀▀ █ █▄░█ █▀▀ █▀▀ █▀█ █▀
            # █▀░ █ █░▀█ █▄█ ██▄ █▀▄ ▄█
            # print(fingers1, fingers2)
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
            print("just predictions", predictions)
            print("ln 394 PREDICTION", max_and_index(predictions[0]))

    cv2.imshow("output", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    saveJsonForUnity()

print("closing json file")
jsonForUnity.close()
print("done!")