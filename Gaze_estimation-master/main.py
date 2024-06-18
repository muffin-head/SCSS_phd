import cv2
import mediapipe as mp

# Assuming gaze.py contains a function named process_gaze that does the gaze estimation
from gaze import process_gaze

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # This includes iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Adjust the camera index based on your setup
cv2.namedWindow('output window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('output window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit(1)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # To improve performance, make the image not writeable

        # Process the image and find face landmarks
        results = face_mesh.process(image)

        # Convert the image color back to BGR to display it correctly
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process each face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                process_gaze(image, face_landmarks)  # Perform gaze estimation

        # Display the resulting frame
        cv2.imshow('output window', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
