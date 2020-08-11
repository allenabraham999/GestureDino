import cv2
from tensorflow.keras.models import model_from_json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

j_file = open("model_json","r")
loaded_model = j_file.read()
j_file.close()

model = model_from_json(loaded_model)
model.load_weights("model.h5")


driver = webdriver.Chrome()
driver.get("http://www.trex-game.skipser.com/")
whole = driver.find_element_by_xpath("/html")

video = cv2.VideoCapture(0)

while(True):
    ret, frame = video.read()
    cv2.rectangle(frame, (10, 10), (200, 200), (0, 0, 225), 3)
    roi = frame[10:200, 10:200]
    frame = cv2.flip(frame, 1)
    roi = cv2.flip(roi, 1)
    roi = cv2.resize(roi,(64,64))
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, roi = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    prediction = model.predict(roi.reshape(1,64,64,1))
    if prediction[0][0] == 0:
        whole.send_keys(Keys.SPACE)
        cv2.putText(frame,"Jumping",(10,100),cv2.FONT_HERSHEY_PLAIN,3,(255,200,10),2)
    elif prediction[0][0] == 1:
        whole.send_keys(Keys.DOWN)
        cv2.putText(frame,"Crawling", (10,120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 200, 10), 2)
    if cv2.waitKey(1) & 0xff==ord("q"):
        break

cv2.destroyAllWindows()
