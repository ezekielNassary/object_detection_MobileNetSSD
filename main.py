import cv2
import imutils
import time
import numpy as np
import asyncio
import websockets
import json
import datetime


threshold = 0.6
classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

counter = 0
width = []
height = []


async def show_img0(img):
    wLabel="W"
    lLabel="L"
    global counter, width, height
    classIds, confs, bbox = net.detect(img, confThreshold=threshold)
    (h, w) = img.shape[:2]

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box.astype('int')
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            label = f'{classNames[classId-1]}: {confidence:.2f}'
            dpi_x = img.shape[0] / (h * 0.393701)
            dpi_y = img.shape[1] / (w * 0.393701)
            dpi = (dpi_x + dpi_y) / 2

           

            physical_width = (w / dpi) * 2.54
            physical_height = (h / dpi) * 2.54

            dpi = 40
            physical_width =await pixels_to_inches(w, dpi)
            physical_height =await pixels_to_inches(h, dpi)
            #print(f"{pixels} pixels is equal to {inches:.2f} inches")

            if classNames[classId-1] == 'bird':
                label="fish"

            if classNames[classId-1] == 'add the object name to perfom an action':

                counter += 1
                width.append(physical_width)
                height.append(physical_height)
                print(counter)
                if counter == 10:
                    cv2.imwrite("./saved_image.png", img)
                    counter=0
                    average_width = sum(width) / len(width)
                    average_height = sum(height) / len(height)
                    print("data sent")
                    # await send_parameters(average_width, average_height)

            cv2.putText(img, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img,f" {physical_height:.2f} cm, {h}",(box[0]+0,box[1]+200), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
            cv2.putText(img, f" {physical_width:.2f} cm , {w}", (box[0] + 100, box[1] + 0), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
async def pixels_to_inches(pixels, dpi):
    inches = pixels / dpi
    cm=inches*2.54
    return cm

async def send_parameters(w,h):
    # ws = websocket.create_connection('ws://localhost:8080')
    
    async with websockets.connect('ws://localhost:8080') as websocket:
        # Send height and width parameters
        image_link="saved_image.png";
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        length=0.0
        if w > h:
            length=w
        else:
            length=h

        data = {"length":f"{length:.2f}","date": formatted_datetime}
        json_data = json.dumps(data)
        await websocket.send(json_data)
        result = websocket.recv()
        print(f'Received result: {result}')
        websocket.close()


async def capture_images():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    while True:
        ret, img = cap.read()
        if ret:
            await show_img0(img)
            cv2.imshow("output 1",img)
            cv2.waitKey(1)
        # await asyncio.sleep(5)


async def main():
    await capture_images()


# start the event loop and run the async loop indefinitely
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
