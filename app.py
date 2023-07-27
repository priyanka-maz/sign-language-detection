import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import io
import base64
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template
from webcam2 import *

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, support_credentials=True)



@socketio.on('image')
def image(data_image):
    
    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    
    img, text = sign_pred(frame, True)
    
    
    frame = cv2.putText(img, 'CV', (220,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA)
    
     # Encode the frame as base64 string
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Emit the frame data back to JavaScript client
    socketio.emit('processed_frame', jpg_as_text)
    

@app.route('/detect', methods=['POST', 'GET'])
def detect():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def landing():
    return render_template('landing.html')

@app.route('/about', methods=['POST', 'GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    socketio.run(app, port = '5000', debug=True)
