import cv2
import numpy as np
from flask import Flask, jsonify
import threading

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Open webcam

# Lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# High frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_face.mp4', fourcc, 24.0, (640, 480))

# Load a more efficient face detection model if available, else use Haar cascades
try:
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
except Exception:
    print("Using Haar Cascades as fallback for face detection.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_and_save_video():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Implement face detection using DNN or Haar based on availability
        if 'net' in globals():
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    cv2.rectangle(mask, (x, y), (x2, y2), (255, 255, 255), -1)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            mask = np.zeros_like(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Apply mask to frame to remove background
        foreground = cv2.bitwise_and(frame, mask)
        out.write(foreground)  # Save the processed frame

    # Cleanup
    cap.release()
    out.release()

@app.route('/start', methods=['GET'])
def start_processing():
    threading.Thread(target=process_and_save_video, daemon=True).start()
    return jsonify({"status": "Processing started"}), 200

@app.route('/stop', methods=['GET'])
def stop_processing():
    if cap.isOpened():
        cap.release()
        out.release()
        return jsonify({"status": "Processing stopped"}), 200
    else:
        return jsonify({"status": "No active processing"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
