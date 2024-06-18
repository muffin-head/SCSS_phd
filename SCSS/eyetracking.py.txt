import cv2
import mediapipe as mp

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

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect face landmarks and iris
    results = face_mesh.process(image_rgb)
    
    # Draw the face mesh and iris annotations on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
            
            # Draw the iris landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            
            # Extract and draw left and right eye landmarks
            left_eye = face_landmarks.landmark[474:478]
            right_eye = face_landmarks.landmark[469:473]
            left_iris = face_landmarks.landmark[468:473]
            right_iris = face_landmarks.landmark[473:478]
            
            for landmark in left_eye:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
            for landmark in right_eye:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
            for landmark in left_iris:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                
            for landmark in right_iris:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    # Display the output
    cv2.imshow('MediaPipe Iris Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
