# https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service
# https://github.com/bradyoo12/helloworld.git

# ssh -i C:\Users\yooho\Downloads\mozaikyou-vm.pem bradyoo@52.141.27.39
# flask run --cert=adhoc 
# 

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

import cv2
from azure.storage.blob import BlobServiceClient
import numpy as np
import os
import datetime
# import mediapipe as mp
from ultralytics import YOLO # pip install ultralytics
from moviepy.editor import (AudioFileClip, VideoFileClip)
# import ffmpeg
from pathlib import Path
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
model = YOLO("best.pt")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_classifier = cv2.CascadeClassifier('cascade.xml')


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask_shape = (img.shape[0], img.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # if faces is ():
    if faces == ():
        print("no face")
        return img
    
    print("found!!")
    temp_img = img

    for (x,y,w,h) in faces:
        temp_img[y:y+h, x:x+w] = cv2.blur(temp_img[y:y+h, x:x+w], (25, 25))

        # create the circle in the mask and in the temp_img, notice the one in the mask is full
        cv2.circle(mask, (int((x + x + w)/2), int((y + y + h)/2)), int(h / 2), (255), -1)

        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img_fg = cv2.bitwise_and(temp_img, temp_img, mask=mask)
    combined = cv2.add(img_bg, img_fg)
    return combined

def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # print(x1, y1, w, h)

            # blur faces
            try:
                img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
            except:
                print("y1 h x1 w:")# + y1 + " " + h + " " + x1 + " " + w)

    return img

def blur_with_YOLO(frame):
    results = model(frame)
    # boxes = results.xyxy[0].numpy()
    for r in results:
        boxes = r.boxes

        for box in boxes:
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            face = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            face = cv2.GaussianBlur(face, (31, 31), 30)

            frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = face
            
    return frame



@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")
        
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadFileForm()
    # if form.validate_on_submit():
    if request.method == 'POST':
        file = request.files['file']
        # filename = secure_filename(file.filename)
        
        # file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        return "File has been uploaded."
    return render_template('upload.html', form=form)

@app.route('/process', methods=['GET', 'POST'])
def process():
# https://hideface.blob.core.windows.net/clip/clip1.mp4   
    url = request.form.get('url')
    cap = cv2.VideoCapture(url)
    filename = os.path.basename(url)
    # cap = cv2.VideoCapture('https://mozaikface.blob.core.windows.net/video/KakaoTalk_20240203_202132549.mp4')
    # cap = cv2.VideoCapture('KakaoTalk_20240203_202132549.mp4')

    now = datetime.datetime.now()
    localmp4 = 'out_' + now.strftime("%Y%m%d-%H%M%S") + '_' + filename
    
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(localmp4, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # mp_face_detection = mp.solutions.face_detection

    #with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9) as face_detection:
    while ret:
            
        frame = blur_with_YOLO(frame)
        
        # frame = detect_faces(frame)
        
        # frame = process_img(frame, face_detection)

        # cv2.imshow('Video face detection', frame)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()

    # audio = AudioFileClip(url)
    # clip = VideoFileClip(localmp4)
    # clip.set_audio(audio)
    
    # finalLocalMp4 = 'audio_' + localmp4
    finalLocalMp4 = localmp4

    # clip.write_videofile(finalLocalMp4)

    # input_audio = ffmpeg.input(url)
    # input_video = ffmpeg.input('./' + localmp4)
    # try:
    #     out1 = ffmpeg.output(input_video, input_audio, './' + finalLocalMp4, vcodec='copy', acodec='aac', strict='experimental')
    #     out1.run()        
    # except:
    #     print('error!')

    # Create the BlobServiceClient object
    # blob_service_client = BlobServiceClient.from_connection_string('DefaultEndpointsProtocol=https;AccountName=mozaikyouvm;AccountKey=om3u1acvei9lyz9AYXVLVKsriSxCS9M1V8KOloM2KQte2MoDhuBAtaz0KnhzPgmALc3uvzsmTAL2+AStxiat6g==;EndpointSuffix=core.windows.net') # (account_url, credential=default_credential)
    blob_service_client = BlobServiceClient.from_connection_string('DefaultEndpointsProtocol=https;AccountName=mozaikface;AccountKey=fnw7p7W3/TvZtbhaECo7ehynwioDf9qRm8DuWUbEQAfC9oErfnP/Wk5tdIFD/8jz4RpwL6xWYTEe+AStJ82neA==;EndpointSuffix=core.windows.net') # (account_url, credential=default_credential)
    
    # container_client = blob_service_client.create_container('video')
    blob_client = blob_service_client.get_blob_client(container='video', blob=finalLocalMp4)
    with open(file=finalLocalMp4, mode="rb") as data:
        blob_client.upload_blob(data)

    os.remove(str(Path(__file__).parent.absolute()) + '/' + finalLocalMp4)
    
    # try:
    #     file_path.unlink()
    # except FileNotFoundError:
    #     print(f"Error: cannot find the '{file_path}' file.")
    # except PermissionError:
    #     print(f"Error: lack permission to delete '{file_path}'.")
    # else:
    #     print(f"Success! Removed '{file_path.relative_to(Path.cwd())}' "
    #         f"from '{Path.cwd()}'.")
    return render_template('process.html', url = blob_client.url)

if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port=8080)
   # app.run(ssl_context='adhoc',debug=True, host="0.0.0.0", port=8080)