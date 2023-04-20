from pickle import TRUE
import cv2
import mediapipe as mp
import time

class PoseEstimation:
    # def __init__(self,mode=False,upper=False,smooth=True,detectCon=0.5,trackCon=0.5):        
    def __init__(self):        
        # self.mode = mode
        # self.upper = upper
        # self.detectCon = detectCon
        # self.trackCon = trackCon
        # self.smooth = self.smooth
        
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        
        # self.pose = self.mpPose.Pose(self.mode,self.upper,self.smooth,self.detectCon,self.trackCon)
        self.pose = self.mpPose.Pose()
    
    def findPose(self,img,draw =True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        self.rlm = results.pose_landmarks
        
        return img
def main():
    pTime = 0 
    cap = cv2.VideoCapture()
    pose = PoseEstimation()
    while True:
        succes,img = cap.read()
        img = pose.findPose(img)
        
        cTime = time.time()
        fps =1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                
        cv2.imshow("Image",img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # if key == ord("s"):
        #     ps =detector.findHands(img)
        #     print(ps)
            
if __name__ == '__main__':
    main()