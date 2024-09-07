import os
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

#variables
width, height = 1300, 720
folderPath = "presentation"

#camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#get the list of presentation Images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

#variables
imgNumber = 0
hs, ws =int(120*1), int(213*1)
gestureThreshold = 350
buttonPressed= False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

#hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    #Import Images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width,gestureThreshold),(0, 255, 0), 10)


    if hands and buttonPressed is False:
        # annotationStart = False
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        print(fingers)
        lmList = hand['lmList']

        if cy <= gestureThreshold:
            if cy >= int(np.interp(lmList[8][0], [height // 2, w], [0, width])):#if hand is at the height of the face
                annotationStart = False

                #Gesture 1- Left
                if fingers == [1, 1, 1, 1, 1]:
                    annotationStart = False
                    print("Left")
                    if imgNumber > 0:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        imgNumber -= 1

        if cy <= gestureThreshold:
            if cy <= int(np.interp(lmList[8][0], [height // 2, w], [0, width])):#if hand is at the height of the face
                annotationStart = False
                if fingers == [1, 1, 1, 1, 1]:
                    annotationStart = False
                    print("Right")
                    if imgNumber < len(pathImages)-1:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        imgNumber += 1




    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        print(fingers)
        lmList = hand['lmList']   #lmList is basically Landmark

        #constrain value for easier drawing

        xVal = int(np.interp(lmList[8][0], [width // 2, w], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        # if cy <= gestureThreshold: #if hand is at the height of the face
        #     annotationStart = False

            #Gesture 1- Left
            # if fingers == [1, 0, 0, 0, 0]:
            #     annotationStart = False
            #     print("Left")
            #     if imgNumber > 0:
            #         buttonPressed = True
            #         annotations = [[]]
            #         annotationNumber = 0
            #         imgNumber -= 1

            # Gesture 2- right
            # if fingers == [1, 1, 1, 1, 1]:
            #     annotationStart = False
            #     print("Right")
            #     if imgNumber < len(pathImages)-1:
            #         buttonPressed = True
            #         annotations = [[]]
            #         annotationNumber = 0
            #         imgNumber += 1


        #Gesture 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart =False

        # Gesture 4- draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber +=1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 8, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5- Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotations:
                    if annotationNumber >=0:
                        annotations.pop(-1)
                        annotationNumber -= 1
                        buttonPressed = True

    else:
        annotationStart = False


    #button Pressed itteration
    if buttonPressed:
        buttonCounter +=1
        if buttonCounter> buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range (len(annotations)):
        for j in range (len(annotations[i])):
            if j != 0 :
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 10)

    #adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w- ws:w] = imgSmall


    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break