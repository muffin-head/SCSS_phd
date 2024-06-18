import cv2
import mediapipe as mp
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize MediaPipe Face Mesh and Iris
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Initialize Face Mesh with Iris tracking
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # This option enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Use a Tkinter dialog to choose an image file
Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
image_path = askopenfilename()  # Show an "Open" dialog box and return the path to the selected file

if image_path:
    background_image = cv2.imread(image_path)
    if background_image is None:
        print("Failed to load image.")
    else:
        background_image = cv2.resize(background_image, (1920, 1080))  # Resize to fullscreen resolution
        cv2.namedWindow('MediaPipe Iris Tracking', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Iris Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Start capturing video from the webcam
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (1920, 1080))  # Resize webcam feed to match the background image size
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image to detect face landmarks and iris
            results = face_mesh.process(image_rgb)
            
            # Overlay the webcam image on the background image
            overlay_image = cv2.addWeighted(background_image, 0.2, image, 0.8, 0)

            # Draw the face mesh and iris annotations on the overlay image
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=overlay_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
                    mp_drawing.draw_landmarks(
                        image=overlay_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec)
            
            # Display the output
            cv2.imshow('MediaPipe Iris Tracking', overlay_image)
            
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing the escape key
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    print("No image selected!")
