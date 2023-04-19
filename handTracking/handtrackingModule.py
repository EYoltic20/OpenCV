from json.tool import main
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode = False, max_hands = 2, detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon        
        self.mpHans = mp.solutions.hands
        

        self.hands =  self.mpHans.Hands(self.mode,self.max_hands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.rhm = self.results.multi_hand_landmarks
        
        if self.rhm:
            for handlms in self.rhm:
                if draw:
                     self.mpDraw.draw_landmarks(img,handlms,self.mpHans.HAND_CONNECTIONS)  
                     
        return img    
 
    def findPosition(self,img,handNo= 0 ,draw=True):
        lmList=[]
        if self.rhm:
            handTosearh = self.rhm[handNo]
            for id  , lm in enumerate(self.handlms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),13,(255,0,240),cv2.FILLED)
        return lmList

def main():
    pTime = 0 
    cTime = 0
    cap= cv2.VideoCapture(0)
    detector =  handDetector()
    while True:
        succes, img = cap.read()
        img = detector.findHands(img)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                
        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            ps =detector.findHands(img)
            print(ps)
            
if __name__ == '__main__':
    main()