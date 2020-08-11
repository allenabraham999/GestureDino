import os
import cv2
import keyboard
import time
#creating the file paths to save our images for training later on
if not os.path.exists("Data"):
    os.makedirs("Data")
    os.makedirs("Data/up")
    os.makedirs("Data/down")

vid = cv2.VideoCapture(0)
count ={"up":len(os.listdir("Data/up")),
        "down":len(os.listdir("Data/down"))}

while(True):
    ret,frame = vid.read()
    cv2.rectangle(frame,(10,10),(200,200),(0,0,225),3)
    roi = frame[10:200,10:200]
    frame = cv2.flip(frame, 1)
    roi = cv2.flip(roi,1)

    cv2.putText(frame,f"UP:{count['up']}",(10,100),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0))
    cv2.putText(frame, f"DOWN:{count['down']}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
    
    roi = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
    _,roi = cv2.threshold(roi,210,255,cv2.THRESH_BINARY)
    
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)

    if keyboard.is_pressed(keyboard.KEY_UP):
        cv2.imwrite("Data/up/"+str(count["up"])+".jpg",roi)
        count["up"]+=1
        time.sleep(1) 
    if keyboard.is_pressed(keyboard.KEY_DOWN):
        cv2.imwrite("Data/down/"+str(count["down"])+".jpg",roi)
        count["down"]+=1
        time.sleep(1)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()


