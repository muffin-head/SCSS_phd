from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
from gaze import process_gaze, generate_heatmap, reset_timers

app = Flask(__name__)

# Initialize and configure MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper function to generate frame by frame from camera
def gen_frames():  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        raise Exception("Error: Camera could not be opened.")

    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            window_width, window_height = 540, 540
            num_grids = 3
            grid_size = (window_width // num_grids, window_height // num_grids)
            grid_counts = [[0]*3 for _ in range(3)]
            grid_timers = [[time.time()]*3 for _ in range(3)]
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    image, grid_counts, grid_timers = process_gaze(image, face_landmarks, grid_size, grid_counts, grid_timers)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
