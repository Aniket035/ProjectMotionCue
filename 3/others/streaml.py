import cv2
import autopy
import time
import numpy as np
import math
import mediapipe as mp
import streamlit as st

############################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
############################


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []
        self.wScr, self.hScr = autopy.screen.size()
        self.plocX, self.plocY = 0, 0

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    self.lmList = lmList
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        bbox = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            xList = [lm[1] for lm in lmList]
            yList = [lm[2] for lm in lmList]
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)

        return lmList, bbox

    def fingersUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            tipIds = [4, 8, 12, 16]
            fingers = []
            for id in tipIds:
                if hand.landmark[id].y < hand.landmark[id - 1].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers

    def findDistance(self, p1, p2, img):
        if self.lmList:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            return length, img, [x1, y1, x2, y2, cx, cy]
        else:
            return None, img, None


def main():
    pTime = 0

    detector = HandDetector(detectionCon=0.5, trackCon=0.5)

    st.title("Hand Tracking App")
    st.write("Click the button below to start tracking your hand.")

    start_button = st.button("Start Tracking")
    placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(4, hCam)

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                fingers = detector.fingersUp()

                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                              (255, 0, 255), 2)

                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (frameR, wCam-frameR), (0, detector.wScr))
                    y3 = np.interp(y1, (frameR, hCam-frameR), (0, detector.hScr))

                    clocX = x3
                    clocY = y3

                    autopy.mouse.move(clocX, clocY)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    detector.plocX, detector.plocY = clocX, clocY

                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(8, 12, img)
                    if length < 39:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]),
                                   15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            placeholder.image(img, channels="BGR", use_column_width=True)


if __name__ == "__main__":
    main()
