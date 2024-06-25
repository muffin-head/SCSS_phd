import cv2
import mediapipe as mp
from gaze import process_gaze

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Fixed window size
window_width, window_height = 540, 540  # Set your desired dimensions

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('output window', cv2.WINDOW_NORMAL)  # Changed to WINDOW_NORMAL for fixed size
cv2.resizeWindow('output window', window_width, window_height)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit(1)

# Grid configuration
num_grids = 10  # For example, a 10x10 grid
grid_size = (window_width // num_grids, window_height // num_grids)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improve performance

        # Process the image and find face landmarks
        results = face_mesh.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process each face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                process_gaze(image, face_landmarks, grid_size)  # Pass grid size

        # Display the resulting frame
        cv2.imshow('output window', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
