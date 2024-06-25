import cv2
import numpy as np
from helpers import relative, relativeT

def process_gaze(frame, points, grid_size, sensitivity=2.0, max_gaze_range=0.5):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    Adjusted to maximize screen reach by modifying how gaze displacement is calculated.
    """
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)  # Screen center

    # Get pupil locations
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)
    sensitivity=10
    # Calculate the vector from the center to each pupil and scale it by the sensitivity
    left_vector = np.array([left_pupil[0] - center[0], left_pupil[1] - center[1]]) * sensitivity
    right_vector = np.array([right_pupil[0] - center[0], right_pupil[1] - center[1]]) * sensitivity

    # Calculate new gaze points by applying these vectors to the screen center
    left_gaze_point = (int(center[0] + left_vector[0]), int(center[1] + left_vector[1]))
    right_gaze_point = (int(center[0] + right_vector[0]), int(center[1] + right_vector[1]))

    # Average the gaze points to get a final gaze direction
    final_gaze_x = (left_gaze_point[0] + right_gaze_point[0]) // 2
    final_gaze_y = (left_gaze_point[1] + right_gaze_point[1]) // 2

    # Adjust for grid if necessary
    grid_x, grid_y = grid_size
    final_gaze_point = (
        final_gaze_x // grid_x * grid_x + grid_x // 2,
        final_gaze_y // grid_y * grid_y + grid_y // 2
    )

    # Draw lines from each pupil to the final gaze point for visual feedback
    cv2.line(frame, left_pupil, final_gaze_point, (255, 0, 0), 2)  # Red for left eye
    cv2.line(frame, right_pupil, final_gaze_point, (0, 255, 0), 2)  # Green for right eye

    return frame