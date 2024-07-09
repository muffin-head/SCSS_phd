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

# Load background image
background_image = cv2.imread('C:\\Users\\c23005186\\Downloads\\phd\\Gaze_estimation-master-bk\\Gaze_estimation-master\\imageui.png')  # Specify the path to your background image
if background_image is None:
    print("Failed to load image.")
    exit(1)
background_image = cv2.resize(background_image, (window_width, window_height))

# Ensure background image is in BGR
if background_image.shape[2] == 4:  # Assuming background_image could be RGBA
    background_image = cv2.cvtColor(background_image, cv2.COLOR_RGBA2BGR)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('output window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output window', window_width, window_height)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit(1)

# Grid configuration
num_grids = 3
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

        # Resize image to match background dimensions (just in case)
        image = cv2.resize(image, (window_width, window_height))

        # Overlay the webcam image on the background image
        overlay_image = cv2.addWeighted(background_image, 0.2, image, 0.8, 0)

        # Process each face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                process_gaze(overlay_image, face_landmarks, grid_size)  # Use overlay_image for processing

        # Display the resulting frame
        cv2.imshow('output window', overlay_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
