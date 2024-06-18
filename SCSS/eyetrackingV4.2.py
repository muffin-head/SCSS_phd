import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize MediaPipe Face Mesh and Iris
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pygame setup for display
pygame.init()
screen_size = (1280, 720)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Gaze Calibration and Tracking')

# Function to draw a calibration point with a fixation cross
def draw_calibration_point(screen, point):
    screen.fill((0, 0, 0))  # Clear screen
    pygame.draw.circle(screen, (255, 0, 0), point, 20)  # Draw larger circle
    pygame.draw.line(screen, (255, 255, 255), (point[0] - 10, point[1]), (point[0] + 10, point[1]), 2)  # Horizontal line
    pygame.draw.line(screen, (255, 255, 255), (point[0], point[1] - 10), (point[0], point[1] + 10), 2)  # Vertical line
    pygame.display.flip()

# Function to capture iris center from webcam
def capture_iris_center(face_mesh, cap):
    success, image = cap.read()
    if not success:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            iris_left = face_landmarks.landmark[474]  # Index for left iris
            iris_right = face_landmarks.landmark[469]  # Index for right iris
            iris_center = (
                (iris_left.x + iris_right.x) / 2 * screen_size[0],
                (iris_left.y + iris_right.y) / 2 * screen_size[1]
            )
            return (int(iris_center[0]), int(iris_center[1]))
    return None

# Perform calibration with more points and improved focus
def perform_calibration(calibration_points):
    cap = cv2.VideoCapture(0)
    calibration_data = {}
    pygame.display.flip()  # Ensure the screen is updated before starting

    for point in calibration_points:
        draw_calibration_point(screen, point)
        pygame.time.wait(2000)  # Wait for 2 seconds to allow the user to focus

        iris_positions = []
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 2000:  # Capture for 2 seconds
            iris_position = capture_iris_center(face_mesh, cap)
            if iris_position:
                iris_positions.append(iris_position)

        if iris_positions:
            avg_iris_position = np.mean(iris_positions, axis=0)
            calibration_data[point] = (int(avg_iris_position[0]), int(avg_iris_position[1]))

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return calibration_data

    cap.release()
    return calibration_data

# Simple geometric mapping function
def map_iris_to_screen(iris_position, calibration_data):
    if not calibration_data or not iris_position:
        return (640, 360)  # Return center as default if no data available or if iris_position is None

    # Calculate weighted sum of screen points based on the inverse distance to each calibration iris position
    weighted_x = 0
    weighted_y = 0
    total_weight = 0

    for screen_point, iris_point in calibration_data.items():
        # Calculate the inverse Euclidean distance to use as weight
        distance = np.sqrt((iris_position[0] - iris_point[0])**2 + (iris_position[1] - iris_point[1])**2)
        if distance == 0:
            return screen_point  # Return immediately if exactly on a calibration point
        weight = 1 / distance
        weighted_x += screen_point[0] * weight
        weighted_y += screen_point[1] * weight
        total_weight += weight

    # Calculate weighted average
    if total_weight == 0:
        return (640, 360)  # Avoid division by zero, return center
    average_x = int(weighted_x / total_weight)
    average_y = int(weighted_y / total_weight)

    return (average_x, average_y)

# Main execution
calibration_points = [(100, 100), (1180, 100), (100, 620), (1180, 620), (640, 360), (0, 0), (1280, 720)]
calibration_data = perform_calibration(calibration_points)

# Validation step: Ask user to look at few points again to check accuracy
validation_points = [(640, 360), (100, 100), (1180, 620)]
print("Validation Results:")
cap = cv2.VideoCapture(0)
for point in validation_points:
    draw_calibration_point(screen, point)
    pygame.time.wait(2000)  # Wait for 2 seconds to allow the user to focus
    iris_position = capture_iris_center(face_mesh, cap)
    if iris_position:
        predicted_position = map_iris_to_screen(iris_position, calibration_data)
        print(f"Expected: {point}, Detected: {predicted_position}")
cap.release()

# Clean up and exit
pygame.quit()
