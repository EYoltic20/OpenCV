import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture(0)

mpHans = mp.solutions.hands

hands =  mpHans.Hands()

mpDraw = mp.solutions.drawing_utils


pTime = 0 
cTime = 0

while True:
    colors = {1:(240,0,255),
              2:(255,0,240),
              3:(255,100,255),
              4:(255,200,255),
              5:(255,300,255),
              6:(255,0,0),
              7:(0,0,255),
              8:(10,100,255),
              9:(100,100,100),
              10:(240,0,255),
              11:(255,0,240),
              12:(255,100,255),
              13:(255,200,255),
              14:(255,300,255),
              15:(255,0,0),
              16:(0,0,255),
              17:(10,100,255),
              18:(100,100,100),
              19:(0,0,255),
              20:(10,100,255),
              0:(100,100,100),
    } 
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    rhm = results.multi_hand_landmarks
    
    if rhm:
        for handlms in rhm:
            for id  , lm in enumerate(handlms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h) 
                cv2.circle(img,(cx,cy),20,colors[id],cv2.FILLED)
                
                
            mpDraw.draw_landmarks(img,handlms,mpHans.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
    