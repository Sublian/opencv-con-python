from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def video_con_opencv():
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n'+bytearray(encodedImage)+b'\r\n')

#rutas
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/video_feed")
def video_feed():
    return Response(video_con_opencv(),mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    app.run(debug=True)

cap.release()