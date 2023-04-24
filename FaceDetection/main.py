import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
font  = cv2.FONT_HERSHEY_PLAIN

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mpFaceDetection.FaceDetection(0.8)


pTime = 0
while True:
    succes,img = cap.read()
     
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = mpFaceDetection.process(imgRGB) 
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox =  int(bboxC.xmin * iw) , int(bboxC.ymin*ih),int(bboxC.width * iw) , int(bboxC.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2,)
            cv2.putText(img,f'score:{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),font,3,(255,0,0),3)
            
            
            
    
    cv2.putText(img,str(int(fps)),(10,70),font,3,(255,0,0),3)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break