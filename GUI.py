from flask import Flask,render_template,Response
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
md = load_model('pranto.h5')
currency = ["1", "10", "100", "1000", "2", "20", "200", "5", "50", "500"]
app=Flask(__name__)
camera=cv2.VideoCapture(0)
def generate_frames():
    count = 1
    while True:
            
        ## read the camera frame...... Frames are numpy arrays but doesn't work with tensorflow.
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            cv2.imwrite('testFile.png',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if count%100==0:
                image_size = (250, 120)
                img = keras.preprocessing.image.load_img('F:/Pictures/testFile.png', target_size=image_size)
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = md.predict(img_array)
                print(currency[predictions.argmax()])
                count = 1
            else:
                count = count + 1
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)