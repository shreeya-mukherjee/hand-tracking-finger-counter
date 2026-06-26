import cv2
import time
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    if not success:
        continue

    img = detector.findHands(img)

    totalFingers = 0

    if detector.results.multi_hand_landmarks:

        numHands = len(detector.results.multi_hand_landmarks)

        for handNo in range(numHands):

            lmList = detector.findPosition(img, handNo, draw=False)

            if len(lmList) != 0:

                fingers = []

                # Thumb
                if handNo == 0:
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Other 4 fingers
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers += fingers.count(1)

    cv2.rectangle(img, (20, 225), (170, 425),
                  (0, 255, 0), cv2.FILLED)

    cv2.putText(img,
                str(totalFingers),
                (45, 375),
                cv2.FONT_HERSHEY_PLAIN,
                10,
                (255, 0, 0),
                25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)

    pTime = cTime

    cv2.putText(img,
                f'FPS: {int(fps)}',
                (400, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 0),
                3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()