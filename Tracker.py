from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from collections import defaultdict
from scipy.stats import gaussian_kde
import torch
import cv2
import numpy as np

# Define a color palette
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# Configure the DeepSort tracker using a YAML file
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=False)

# Global dictionary to store the trajectories of each track
track_history = defaultdict(list)
track_visualization = defaultdict(list)

# Function to calculate moving average of object trajectories
def trajectory_moving_average(values, window_size):
    if len(values) < window_size:
        return values[-1]

    sum_x = sum(point[0] for point in values[-window_size:])
    sum_y = sum(point[1] for point in values[-window_size:])
    avg_x = sum_x / window_size
    avg_y = sum_y / window_size

    return (int(avg_x), int(avg_y))

# Frame buffer for moving average calculation
frame_buffer = []
buffer_size = 5

# Function to apply density estimation to an image using Gaussian Kernel Density Estimation
def apply_density_estimation(image, points):
    try:
        values = np.vstack(points)
        kernel = gaussian_kde(values)

        # Evaluate on a grid
        xgrid = np.linspace(0, image.shape[1], image.shape[1])
        ygrid = np.linspace(0, image.shape[0], image.shape[0])
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        grid_coords = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
        Z = np.reshape(kernel(grid_coords), Xgrid.shape)

        Z_normalized = Z / np.max(Z)

        # Convert the normalized density to a heatmap
        heatmap = np.uint8(255 * Z_normalized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlayed_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        # Scale up for visibility
        high_density_area = np.max(Z * 100000)

        cv2.putText(overlayed_image, f'Density: {high_density_area:.2f}', (20, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return overlayed_image
    # Return original frame if can not conduct KDE
    except np.linalg.LinAlgError as e:
        print(f"Skipping frame due to error: {e}")
        return image
    except ValueError as e:
        print(f"Skipping frame due to ValueError: {e}")
        return image

# Function to average frames for motion detection
def moving_average(frame):
    frame_buffer.append(frame)
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    avg_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
    return avg_frame

# Function to detect motion between the current and average frames

def detect_motion(current_frame, avg_frame, threshold=30):
    diff = cv2.absdiff(current_frame, avg_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

# Function to visualize detected motion regions in the image
def draw_motion_regions(image, motion_mask):
    # Find contours from the motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for all motion regions
    all_motion_mask = np.zeros_like(motion_mask)

    # Fill the contour in the mask
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            cv2.drawContours(all_motion_mask, [contour], -1, 255, -1)

    masked_area = cv2.bitwise_and(image, image, mask=all_motion_mask)
    colored_area = cv2.applyColorMap(cv2.cvtColor(masked_area, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    np.copyto(image, colored_area, where=all_motion_mask[:, :, None].astype(bool))

# Function to handle bounding boxes, trajectories, and motion alerts in the image
def draw_bounding_boxes_and_trajectories(image, bounding_boxes, motion_mask, thickness=None):
    line_thickness = thickness or (round(0.002 * (sum(image.shape[:2]) / 2)) + 1)
    average_window = 20
    alert_triggered = False

    for bbox in bounding_boxes:
        x1, y1, x2, y2, category_id, object_id = bbox

        # Extract the portion of the motion mask that overlaps with the bounding box
        box_mask = motion_mask[y1:y2, x1:x2]
        motion_within_box = np.count_nonzero(box_mask)

        motion_threshold = 0.1 * box_mask.size

        if 0.3 * box_mask.size > motion_within_box > motion_threshold:
            color = (0, 255, 0)  # Green if modest motion is detected
        elif motion_within_box > 0.3 * box_mask.size:
            color = (0, 0, 255) # Red if significant motion detected
            alert_triggered = True
        else:
            color = (250, 5, 5)  # blue otherwise

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Draw trajectory
        if object_id in track_visualization:
            for i in range(1, len(track_visualization[object_id])):
                pt1 = track_visualization[object_id][i - 1]
                pt2 = track_visualization[object_id][i]
                cv2.line(image, pt1, pt2, color, thickness=2)

        # Update the current position in track history
        current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        track_history[object_id].append(current_center)

        avg_center = trajectory_moving_average(track_history[object_id], average_window)
        track_visualization[object_id].append(avg_center)

        if alert_triggered:
            alert_text = "Alert: Significant motion detected!"
            cv2.putText(image, alert_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        text = f'{object_id}'
        font_scale = line_thickness / 3
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)[0]
        text_bg_top_left = x1, y1 - text_size[1] - 3
        text_bg_bottom_right = x1 + text_size[0], y1
        cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (x1, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=[225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

    return image

# Main function to update tracking and rendering based on detections
def update_tracker(detector, image_frame, track_ids_to_follow, Density):
    avg_frame = moving_average(image_frame.copy())
    motion_mask = detect_motion(image_frame, avg_frame)
    draw_motion_regions(image_frame, motion_mask)

    detected_bboxes = detector.detect(image_frame)

    bbox_center_wh = []
    detection_confs = []
    detection_classes = []
    points = []

    for (xmin, ymin, xmax, ymax, class_id, confidence) in detected_bboxes:
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)
        width = xmax - xmin
        height = ymax - ymin
        bbox_center_wh.append([center_x, center_y, width, height])
        detection_confs.append(confidence)
        detection_classes.append(class_id)
        points.append([center_x, center_y])

    if points:
        points = np.array(points).T  # Transpose to match format

    tensor_bboxes = torch.Tensor(bbox_center_wh)
    tensor_confs = torch.Tensor(detection_confs)

    tracking_results = deepsort.update(tensor_bboxes, tensor_confs, detection_classes, image_frame)

    prepared_bboxes = []
    for track_result in tracking_results:
        x1, y1, x2, y2, cls_id, track_id = track_result
        prepared_bboxes.append((x1, y1, x2, y2, cls_id, track_id))

    # Filter the prepared bboxes based on the selected track IDs or track all if None
    if track_ids_to_follow is not None:
        prepared_bboxes = [bbox for bbox in prepared_bboxes if bbox[5] in track_ids_to_follow]

    updated_frame = draw_bounding_boxes_and_trajectories(image_frame, prepared_bboxes, motion_mask)

    # Only apply density estimation if Density = 1
    if Density == 1:
        updated_frame = apply_density_estimation(updated_frame, points)

    return updated_frame


