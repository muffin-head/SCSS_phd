import cv2
import numpy as np
from helpers import relative, relativeT  # Ensure these helper functions are correctly defined to use landmarks

def process_gaze(frame, points, grid_size, grid_counts, sensitivity=8):
    """
    Processes the gaze by identifying eye positions and calculating gaze points based on detected face landmarks,
    and maintains count of gazes falling into each of the three grid sections.
    """
    # Assuming the helpers return tuples of (x, y) coordinates
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    left_vector = np.array([left_pupil[0] - center[0], left_pupil[1] - center[1]]) * sensitivity
    right_vector = np.array([right_pupil[0] - center[0], right_pupil[1] - center[1]]) * sensitivity

    left_gaze_point = (int(center[0] + left_vector[0]), int(center[1] + left_vector[1]))
    right_gaze_point = (int(center[0] + right_vector[0]), int(center[1] + right_vector[1]))

    final_gaze_x = (left_gaze_point[0] + right_gaze_point[0]) // 2
    final_gaze_y = (left_gaze_point[1] + right_gaze_point[1]) // 2

    grid_x = final_gaze_x // grid_size[0]
    grid_y = final_gaze_y // grid_size[1]

    if 0 <= grid_x < len(grid_counts[0]) and 0 <= grid_y < len(grid_counts):
        grid_counts[grid_y][grid_x] += 1  # Update the gaze count in the corresponding grid

    cv2.line(frame, left_pupil, (final_gaze_x, final_gaze_y), (255, 0, 0), 2)
    cv2.line(frame, right_pupil, (final_gaze_x, final_gaze_y), (0, 255, 0), 2)

    return frame, grid_counts

def generate_heatmap(grid_counts, frame_size):
    """
    Generates a heatmap from the grid counts.
    """
    heatmap = np.array(grid_counts, dtype=np.float32)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap = cv2.resize(heatmap, frame_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Heatmap', heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('heatmap.png', heatmap)  # Optionally save the heatmap
