from xml.sax.handler import feature_string_interning
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import random
import telepot
model = pickle.load(open('Naive_Bayes.pkl', 'rb'))

import cv2
import os
import time
# Import numpy for matrices calculations
import numpy as np
        
# Create Local Binary Patterns Histograms for face recognization
#recognizer = cv2.face.createLBPHFaceRecognizer()
#recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('FACE_RECOGNITION/trainer/trainer.yml')
##recognizer.read('/home/pi/Desktop/face_recog_folder/Raspberry-Face-Recognition-master/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "FACE_RECOGNITION/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

app = Flask(__name__)

@app.route('/')
def index():
    import random
    a=[]
    for i in range(0,28):
        a.append('{:.4f}'.format(random.uniform(0.0, 1.5)))
    return render_template('index.html', a=a)

@app.route('/', methods=['GET', 'POST'])
def predict():
    import random
    a=[]
    for i in range(0,28):
        a.append('{:.4f}'.format(random.uniform(0.0, 1.5)))
        
    if request.method == 'POST':
        Data=[]
        for i in request.form.listvalues():
            for ii in i:
                Data.append(float(ii))
        print(Data)
        out = model.predict([Data])
        print(out)
        output=''
        if out[0] == 1 or Data[0] <= 110:
            
            output='Fraud Trancaction'
        else:
            
            output='Normal'

        print(output)
        return render_template('home.html', msg = output, a=a)
    return render_template('index.html', a=a)

@app.route('/otp', methods=['GET', 'POST'])
def otp():
    import random
    a=[]
    for i in range(0,28):
        a.append('{:.4f}'.format(random.uniform(0.0, 1.5)))
    if request.method == 'POST':
        num = random.randint(1111, 9999)
        print(num)
        bot = telepot.Bot('5604485306:AAFGv1JwuxcV--KESSs6GQwmpqSwA2OUQaw')
        bot.sendMessage('5750164366', str(num))
        with open('file.txt', 'w') as f:
            f.write(str(num))
            f.close()
        return render_template('home.html', otp=num, a=a)
    return render_template('index.html', a=a)

@app.route('/verification', methods=['GET', 'POST'])
def verification():
    import random
    a=[]
    for i in range(0,28):
        a.append('{:.4f}'.format(random.uniform(0.0, 1.5)))
        
    if request.method == 'POST':
        OTP = int(request.form['OTP'])
        print(OTP)
        with open('file.txt', 'r') as f:
            OTP1 =  f.read()
            f.close()

        print(OTP1)
        OTP1 = int(OTP1)
        
        if OTP == OTP1:
            # Initialize and start the video frame capture
            cam = cv2.VideoCapture(0)

            count = 0
            count1 = 0
            count2 = 0
            while True:

                    # Read the video frame
                    ret, im =cam.read()

                    # Convert the captured frame into grayscale
                    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

                    # Get all face from the video frame
                    faces = faceCascade.detectMultiScale(gray, 1.2,5)

                    # For each face in faces
                    for(x,y,w,h) in faces:
                        count += 1
                        # Recognize the face belongs to which ID
                        Id,i = recognizer.predict(gray[y:y+h,x:x+w])
                        #id = int(os.path.split(imagePath)[-1].split(".")[1])
                        
                        print(i)
                        Id1=''
                        # Check the ID if exist
                        if i < 60:
                            count1 += 1
                            if Id == 1 :
                                Id1 = "vishnu"
                                print(Id1)
                            if Id == 2 :
                                Id1 = "person 2"
                                print(Id1)
                            if Id == 3 :
                                Id1 = "person 3"
                                print(Id1)
                            if Id == 4 :
                                Id1 = "person 4"
                                print(Id1)
                        else:
                            count2 += 1
                            Id1 = "unknown"
                            print(Id1)
                        
                        # Put text describe who is in the picture
                        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                        cv2.putText(im, str(Id1), (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

                    # Display the video frame with the bounded rectangle
                    cv2.imshow('im',im)
                    # If 'q' is pressed, close program
                    if cv2.waitKey(1) & count == 100: #if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                       
            cam.release()
            # Close all windows
            cv2.destroyAllWindows()

            if count1 > count2:
                return render_template('index.html', msg1='face authenticated and transaction Successfull', a=a)
            else:
                return render_template('index.html', msg2='face recognition faild', a=a)
        else:
            return render_template('index.html', msg2='Entered wrong otp', a=a)
        
    return render_template('index.html', a=a)

if __name__ == "__main__":
    app.run(debug=True)
