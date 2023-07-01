# This code run in CPU so it may be bit slower in case of large weight files.
# Video reference link: https://youtu.be/WgPbbWmnXJ8?t=2236
# Use GPU to get better results
# video link for GPU pytorch:  https://youtu.be/WgPbbWmnXJ8?t=4101


import cv2 as cv
from ultralytics import YOLO
import cvzone   # use to display all the detections labels properly on the top of the detction
import math

import pafy

import numpy as np 
import threading
import urllib.request



#===================================For YOUTUBE video=========================================
# source = "https://youtu.be/YbzX7oGhm8w"

# # Create a pafy object
# video = pafy.new(source)

# # Get the best available video stream
# best_stream = video.getbest()
# stream = best_stream.url

# cap = cv.VideoCapture(stream)
#--------------------------------------------------------------------

#============================For Downloaded video==================================
# stream = 'YOLOv8\The Most Lethal Fighter Jet Ever Built _ F-22 Raptor.mp4'
# cap = cv.VideoCapture(stream)
#-------------------------------------------------------------------------

#===============================For ip streams==========================================
url = 'https://ee6c-139-167-217-242.ngrok-free.app/video'


def retrieve_frames(url, output_frames):
    buffer_size = 4096
    bytes =  b''
    stream = urllib.request.urlopen(url)

    while True:
        bytes += stream.read(buffer_size)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        # print(bytes)

        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv.imdecode(np.frombuffer(jpg, dtype= np.uint8), cv.IMREAD_COLOR)
            output_frames.append(frame)

# print("Thread started")

output_frames = []
thread1 = threading.Thread(target=retrieve_frames, args=(url, output_frames))
thread1.daemon = True
thread1.start()
#-----------------------------------------------------------------------------------



# cap.open('http://192.168.43.252/video')
# print (cap.isOpened())
# img_size = cap.shape[:2]
# print(img_size)




# Customize the size of video promt opened of streaming

# cap.set(3, 320)   # set the width of the video box
# cap.set(4, 240)     #set the height of the video box


# Load the YOLO weight file 

model = YOLO('yolov8n.pt')  # load the nano weight file

# Class names

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
              "fire hydrant","stop sign","parking m","bird","cat","dog","horse","sheep","cow","elephant",
              "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
              "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
              "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
              "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]


while True:
    #==================For video captureing===================
    # ignore, img = cap.read()

    # ==================For live stream capturing===================
    if len(output_frames)>0:
        img = output_frames.pop(0)

        results = model(img, stream = True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1, y1, x2, y2)
                cv.rectangle(img, (x1,y1), (x2, y2), (255,0,255), 1)
# Confidence number of detction

            confidence = math.ceil(box.conf[0]*100)/100      # give the confidence value of the bounding box
            # print(confidence) 

# Class Name

            cls = int(box.cls[0])

# Show the class name or id along with the confidence value

            cvzone.putTextRect(img, f'{classNames[cls]}{confidence}', (max(0,x1), max(35, y1)),scale=1, thickness=1)

            cv.imshow("video", img)

        if cv.waitKey(1) & 0xff == ord('d'):
            break
img.release()
cv.destroyAllWindows()
    


        