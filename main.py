# https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service
# https://github.com/bradyoo12/helloworld.git

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

# import cv2
# from azure.storage.blob import BlobServiceClient
# import numpy as np
import os
# import datetime
# # import mediapipe as mp
# from ultralytics import YOLO # pip install ultralytics
# from moviepy.editor import (AudioFileClip, VideoFileClip)
# # import ffmpeg
# from pathlib import Path
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)


@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))