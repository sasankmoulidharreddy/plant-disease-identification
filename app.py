import os
from flask import Flask, redirect, render_template, request, Response, url_for
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import cv2
import datetime
import time
from threading import Thread

global capture, rec_frame, grey, switch, neg, face, rec, out, captured_image_pathrun
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
captured_image_path = ""

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

camera = cv2.VideoCapture(1)

def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, captured_image_path
    while True:
        success, frame = camera.read()
        if success:
            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['static/uploads', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
                captured_image_path = p
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
            while capture == 1:
                time.sleep(0.1)
            return redirect(url_for('submit'))
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    global capture, captured_image_path  # Make sure to access global variables

    if request.method == 'POST':
        image = request.files['image']
        if image:  # Ensure that the image is not None
            filename = image.filename
            file_path = os.path.join('static', 'uploads', filename)
            image.save(file_path)
            pred = prediction(file_path)
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name,
                                   simage=supplement_image_url, buy_link=supplement_buy_link,file_path=file_path)
        else:
            # Handle case where no image is uploaded
            return render_template('index.html', error="No image uploaded")

    elif request.method == 'GET':
        if captured_image_path:
            pred = prediction(captured_image_path)
            filename = captured_image_path
            file_path = os.path.join('static', 'uploads', filename)
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name,
                                   simage=supplement_image_url, buy_link=supplement_buy_link,file_path=file_path)
        else:
            # Handle case where capture is set but image path is not available
            return render_template('index.html', error="Captured image path not set")

    # Default response if no conditions are met
    return render_template('index.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
