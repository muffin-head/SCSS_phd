import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygame
import time

# Initialize MediaPipe Face Mesh and Iris
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing specifications for landmarks and connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Initialize Face Mesh with Iris tracking
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # This option enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Pygame for Radar Display
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption('SCSS Radar Simulation')

# Load Radar Background
radar_image = pygame.image.load('radar_background.png')  # Load your radar background image here
radar_rect = radar_image.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))

# Lists to store iris center coordinates for heatmap
left_iris_centers = []
right_iris_centers = []

def calculate_iris_center(iris_landmarks, image_shape):
    iris_x = [int(landmark.x * image_shape[1]) for landmark in iris_landmarks]
    iris_y = [int(landmark.y * image_shape[0]) for landmark in iris_landmarks]
    center_x = int(np.mean(iris_x))
    center_y = int(np.mean(iris_y))
    return center_x, center_y

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Display the radar image for 10 seconds and capture eye tracking data
start_time = time.time()
running = True
while running and cap.isOpened() and (time.time() - start_time) < 10:
    success, image = cap.read()
    if not success:
        break
    
    # Pygame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect face landmarks and iris
    results = face_mesh.process(image_rgb)
    
    # Draw the face mesh and iris annotations on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right iris landmarks
            left_iris = face_landmarks.landmark[468:473]
            right_iris = face_landmarks.landmark[473:478]
            
            # Calculate the center of the left and right iris
            left_iris_center = calculate_iris_center(left_iris, image.shape)
            right_iris_center = calculate_iris_center(right_iris, image.shape)
            
            # Mirror the x-coordinate to correct the mirroring effect
            left_iris_center = (image.shape[1] - left_iris_center[0], left_iris_center[1])
            right_iris_center = (image.shape[1] - right_iris_center[0], right_iris_center[1])
            
            # Store iris centers for heatmap
            left_iris_centers.append(left_iris_center)
            right_iris_centers.append(right_iris_center)
            
            # Draw the centers of the iris
            cv2.circle(image, left_iris_center, 2, (0, 255, 255), -1)
            cv2.circle(image, right_iris_center, 2, (0, 255, 255), -1)
            
            # Display coordinates
            cv2.putText(image, f"Left Iris: {left_iris_center}", (left_iris_center[0], left_iris_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, f"Right Iris: {right_iris_center}", (right_iris_center[0], right_iris_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Update the Pygame display
    screen.fill((0, 0, 0))  # Clear the screen
    screen.blit(radar_image, radar_rect)  # Draw the radar image in the center
    pygame.display.flip()
    
    # Display the output
    cv2.imshow('MediaPipe Iris Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'Esc'
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()

# Generate heatmaps for left and right iris centers
def generate_heatmap(iris_centers, image_shape, title):
    heatmap, xedges, yedges = np.histogram2d(
        [center[0] for center in iris_centers],
        [center[1] for center in iris_centers],
        bins=(image_shape[1] // 20, image_shape[0] // 20)  # Adjust bins for less grid
    )
    
    # Apply threshold to heatmap values
    heatmap[heatmap < 2] = 0
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap.T, cmap='jet', cbar=True)
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().invert_yaxis()
    plt.show()

# Assuming image_shape is from the last frame captured
image_shape = (720, 1280, 3)  # Modify according to your webcam resolution

# Generate heatmaps
generate_heatmap(left_iris_centers, image_shape, 'Heatmap for Left Iris')
generate_heatmap(right_iris_centers, image_shape, 'Heatmap for Right Iris')
