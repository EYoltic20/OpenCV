import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0
font = cv2.FONT_HERSHEY_PLAIN

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()


while True:
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    rlm = results.pose_landmarks
    if rlm :
        mpDraw.draw_landmarks(img,rlm,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(rlm.landmark):
            h,w,c = img.shape
            # print(id,lm)
            cx,cy = int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
            
            
            
    cTime = time.time()
    fps =1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(fps),(70,50),font,3,(255,0,0),3)
    
    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break