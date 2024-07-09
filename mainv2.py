 # Specify the path to your background image
import cv2
import mediapipe as mp
from gaze import process_gaze, generate_heatmap

# Initialize and configure MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Window and grid setup
window_width, window_height = 540, 540
grid_counts = [[0]*3 for _ in range(3)]

# Load and process background image
background_image = cv2.imread('C:\\Users\\c23005186\\Downloads\\phd\\Gaze_estimation-master-bk\\Gaze_estimation-master\\imageui.png')
background_image = cv2.resize(background_image, (window_width, window_height))
if background_image.shape[2] == 4:
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (window_width, window_height))
        overlay_image = cv2.addWeighted(background_image, 0.2, image, 0.8, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                overlay_image, grid_counts = process_gaze(overlay_image, face_landmarks, grid_size, grid_counts)


        cv2.imshow('output window', overlay_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    generate_heatmap(grid_counts, (window_width, window_height))  # Generate and display the heatmap
