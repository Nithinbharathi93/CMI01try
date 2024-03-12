from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np

def video_feed(request):
    def gen(camera):
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            else:
                # Process the frame here
                processed_frame = process_frame(frame)
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    frame[mask > 0] = (255, 255, 255)
    mask_inv = cv2.bitwise_not(mask)
    frame[mask_inv > 0] = (0, 0, 0)
    return frame

def index(request):
    return render(request, 'CMIApp/index.html')
