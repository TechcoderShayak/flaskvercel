from flask import Flask, render_template, Response,jsonify,request,session,redirect,url_for

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm
import wikipediaapi
from bardapi import Bard
import wget

from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os

import numpy as np


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection

application= Flask(__name__)

application.config['SECRET_KEY'] = 'muhammadmoin'
application.config['UPLOAD_FOLDER'] = 'static/files'

yolov3_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
yolov3_weights_path = "yolov3.weights"

# Download the weights file if it doesn't exist
if not os.path.exists(yolov3_weights_path):
    print("Downloading YOLOv3 weights...")
    wget.download(yolov3_weights_url, yolov3_weights_path)
    print("\nDownload complete!")

net = cv2.dnn.readNet(yolov3_weights_path, 'yolov3.cfg')


@application.route('/test_again')
def test_again():
    return render_template('indexproject.html')


filename = ''
detected_objects = []
classes = []
classname1 = []
class_name = []
output = []
array = []
name = []
topics = []
empty_set = set()
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="app/1.0"  # Replace with your app's name and version
)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
def remove_duplicates(input_array):
    return list(set(input_array))

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")
@application.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        name.clear()
        detected_objects.clear()
        filename = os.path.join(application.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Object Detection
        img = cv2.imread(filename)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                classname1 = classes[class_ids[i]]

                wiki_page = wiki_wiki.page(classname1)
                name.append(label)
                if classname1 not in class_name:
                    class_name.append(classname1)
                    wiki_page = wiki_wiki.page(classname1)
                confidence = confidences[i]

                detected_object = {
                    'class': classname1,
                    'confidence': confidences[i],
                    'box': boxes[i],
                    'summary': wiki_page.summary if wiki_page.exists() else "No Wikipedia data available"
                }
                detected_objects.append(detected_object)

                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        result_image_path = os.path.join(application.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(result_image_path, img)

        return redirect(url_for('uploaded_file', filename='result_' + file.filename))


@application.route('/files/<filename>')
def uploaded_file(filename):
    array.clear()
    output.clear()
    topics.clear()
    filename = filename.split('_')[-1]
    n = remove_duplicates(name)

    return render_template('uploaded.html', filename=filename, name=output, array=array,
                           detected_objects=detected_objects)

def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@application.route('/', methods=['GET','POST'])
@application.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')
# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@application.route("/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('ui.html')
@application.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), application.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), application.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
        
    return render_template('videoprojectnew.html', form=form)
@application.route('/Image')
def image():
    return render_template('index.html')
@application.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@application.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    application.run(debug=True)