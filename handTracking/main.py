from json.tool import main
import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture(0)

mpHans = mp.solutions.hands

hands =  mpHans.Hands()

mpDraw = mp.solutions.drawing_utils


pTime = 0 
cTime = 0



def main():
    while True:
        succes, img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        rhm = results.multi_hand_landmarks
        
        if rhm:
            for handlms in rhm:
                for id  , lm in enumerate(handlms.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h) 
                    cv2.circle(img,(cx,cy),13,(255,0,240),cv2.FILLED)
                    
                    
                mpDraw.draw_landmarks(img,handlms,mpHans.HAND_CONNECTIONS)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        
    

if __name__ =='__main__':
    main()