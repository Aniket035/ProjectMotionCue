from flask import Flask, render_template, request, redirect, url_for, send_file, Response
#import io
#from PIL import Image
#import datetime
#import torch
import cv2
import numpy
import tensorflow as tf
from werkzeug.utils import send_from_directory
#secure_filename
import os 
#import subprocess
#import re
#import requests
#import shutil
import time
#import glob
from ultralytics import YOLO

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_img", methods = ["GET","POST"])    
def predict_img():
    #print("--------------------------------------------------------------------------------------a")
    #print(type(request.method))
    if request.method == "POST":
        print("--------------------------------------------------------------------- yes")
        print(request.files['file'])
        if "file" in request.files:
            print("POST successful")
            f = request.files["file"]
            basepath = os.path.dirname(__file__)
            #print("------------------------------------------------",f.filename)
            filepath = os.path.join(basepath, "uploads", f.filename)
            print("uplaod folder is ",filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict image :: ",predict_img)
            file_extension = f.filename.rsplit(".",1)[1].lower()
            
            if file_extension == "jpg":   #checked
                img = cv2.imread(filepath)

                #frame =cv2.imencode(".jpg",cv2.UMat(img))[1].tobytes()

                #image = Image.open(io.BytesIO(frame))
                #prediction
                yolo = YOLO("best(1).pt") #1
                detections = yolo.predict(img, save=True, conf = 0.5, iou = 0.6, line_width =1)
                return display(f.filename)
            
            elif file_extension == "mp4":
                video_path = filepath # whereever the video is
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter("output.mp4", fourcc,30.0,(frame_width,frame_height))
                model = YOLO("best(1).pt") #2

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    result = model(frame, save=True)
                    print(result)
                    cv2.waitKey(1)

                    res_plotted = result[0].plot(line_width =3)
                    #cv2.imshow("result", res_plotted)

                    out.write(res_plotted)
                    if cv2.waitKey(1) == ord("q"):
                        break
                
                return video_feed()
    
    '''folder_path = "runs/detect"
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key = lambda x: os.path.getctime(os.path.join(folder_path,x)))
    image_path = folder_path+ "/" + latest_subfolder + "/" + f.filename'''
    return render_template("index.html")

@app.route('/display,<path:filename>')    #checked
def display (filename):
    folder_path  = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))] 
    latest_subfolder = max(subfolders, key =lambda x: os.path.getctime(os.path.join(folder_path, x))) 
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print (latest_file)
    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension  = filename.rsplit('.', 1)[1].lower()
    environ = request.environ
    if file_extension =='jpg':
        return send_from_directory(directory, latest_file,  environ)
    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = "output.mp4"
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode(".jpg", image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    print("video feed ")
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/red")
def red():
    return render_template("page.html")

@app.route("/live_camera")
def live_camera():
    return Response(reader(), mimetype="multipart/x-mixed-replace; boundary=frame")


def reader():
    cap = cv2.VideoCapture(0)
    model = YOLO("best(1).pt")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc,30.0,(frame_width,frame_height))
    while True:
        success,frame= cap.read()
        if not success:
            break
        else:
            result = model(frame)
            result_plotted =result[0].plot(line_width =1)
            print(result)
            ret, buffer = cv2.imencode(".jpg", result_plotted)     ##3
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == "__main__":
    app.run(debug=True)
