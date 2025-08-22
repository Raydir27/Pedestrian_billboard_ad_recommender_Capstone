from ultralytics import YOLO
import cv2
import csv
import os
# Assuming these imports and models are correctly set up
from age_gender_detect import load_age_gend_models, detect_age_gender

model_ped = YOLO("../Models/computer_vision_models/pedestrian_detect_model_v8n_24_jul.pt")
model_apparel = YOLO("../Models/computer_vision_models/apparel2_v8n_model_13_aug_2025.pt")

# Load the age/gender models and assign them to variables
faceNet, ageNet, genderNet = load_age_gend_models()

def detect_pedestrian(image):
    results = model_ped(image)
    return results

def detect_apparel(image):
    results = model_apparel(image)
    return results

# predicts both apparel and age/gender
def pedestrian_crop(pedestrian_results, original_image):
    """
    Crops pedestrians from an image based on detection results and runs
    apparel detection on each cropped pedestrian.
    Args:
        pedestrian_results: The results from the detect_pedestrian function.
        original_image: The original image (as a numpy array) that was fed
                        to detect_pedestrian.
    Returns:
        A dictionary where keys are integer IDs (0, 1, 2, ...) for each
        detected pedestrian, and values are another dictionary containing
        the 'bbox' and 'apparel_results'.
    """
    all_detections = {}
    pedestrian_id = 0
    # The result object from YOLO is iterable.
    for result in pedestrian_results:
        # Bounding boxes in xyxy format, moved to CPU and converted to numpy
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            # Extract coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box)
 
            # Crop the pedestrian from the original image
            cropped_pedestrian = original_image[y1:y2, x1:x2]
 
            # Run detection on the cropped image if it's not empty
            if cropped_pedestrian.size > 0:
                apparel_results = detect_apparel(cropped_pedestrian)

                # Run age/gender detection on the cropped image
                # NOTE: The detect_age_gender function is expected to return
                # a dictionary with keys like 'age' and 'gender'
                # and their respective confidence scores.
                age_gender_results = detect_age_gender(cropped_pedestrian, faceNet, ageNet, genderNet)
                
                all_detections[pedestrian_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'apparel_results': apparel_results,
                    'age_gender_results': age_gender_results
                }
                pedestrian_id += 1
 
    return all_detections

# logs the detections from the complete frame to a unique CSV file
# def log_detections_to_csv(detections, csv_filepath, frame_id=0):
#     """
#     Logs apparel and age/gender detections for each pedestrian to a CSV file.

#     Args:
#         detections (dict): The dictionary returned by pedestrian_crop.
#         csv_filepath (str): The path to the output CSV file.
#         frame_id (int): An identifier for the current image or video frame.
#     """
#     # Define the header for the CSV file
#     header = [
#         'frame_id', 'pedestrian_id', 'ped_bbox_x1', 'ped_bbox_y1', 'ped_bbox_x2', 'ped_bbox_y2',
#         'detection_type', 'label', 'confidence',
#         'item_bbox_x1', 'item_bbox_y1', 'item_bbox_x2', 'item_bbox_y2'
#     ]

#     # Check if the file exists to decide whether to write the header
#     file_exists = os.path.isfile(csv_filepath)

#     with open(csv_filepath, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)

#         if not file_exists:
#             writer.writerow(header)

#         # Process each pedestrian's detections
#         for ped_id, data in detections.items():
#             ped_bbox = data['bbox']
#             ped_x1, ped_y1, ped_x2, ped_y2 = ped_bbox

#             # --- Log Apparel Detections (YOLO results) ---
#             # Process apparel results as the original code intended
#             for res in data['apparel_results']:
#                 class_names = res.names
#                 for box in res.boxes:
#                     label = class_names[int(box.cls)]
#                     confidence = float(box.conf)
#                     # item_bbox is relative to the cropped pedestrian image
#                     item_bbox_rel = box.xyxy[0].cpu().numpy().astype(int)
                    
#                     # Convert item bbox to original image coordinates
#                     ix1, iy1, ix2, iy2 = item_bbox_rel
#                     writer.writerow([
#                         frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
#                         'apparel', label, f"{confidence:.4f}",
#                         ped_x1 + ix1, ped_y1 + iy1, ped_x1 + ix2, ped_y1 + iy2
#                     ])

#             # --- Log Age and Gender Detections (List of dictionaries) ---
#             # This section now correctly handles the list of dictionaries from your code snippet.
#             # It also ensures no confidence value is logged for age or gender, as requested.
#             age_gender_results_list = data['age_gender_results']
            
#             for face_detection in age_gender_results_list:
#                 # Log gender
#                 gender_label = face_detection.get('gender')
#                 # No confidence logged for age and gender.
#                 writer.writerow([
#                     frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
#                     'age_gender', gender_label, '',
#                     ped_x1, ped_y1, ped_x2, ped_y2 # Use pedestrian bbox for this entry
#                 ])
                
#                 # Log age
#                 age_label = face_detection.get('age')
#                 # No confidence logged for age and gender.
#                 writer.writerow([
#                     frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
#                     'age_gender', age_label, '',
#                     ped_x1, ped_y1, ped_x2, ped_y2 # Use pedestrian bbox for this entry
#                 ])


def log_detections_to_csv(detections, csv_filepath, frame_id=0):
    """
    Logs apparel and age/gender detections for each pedestrian to a CSV file.

    Args:
        detections (dict): The dictionary returned by pedestrian_crop.
        csv_filepath (str): The path to the output CSV file.
        frame_id (int): An identifier for the current image or video frame.
    """
    # Define the header for the CSV file
    header = [
        'frame_id', 'pedestrian_id', 'ped_bbox_x1', 'ped_bbox_y1', 'ped_bbox_x2', 'ped_bbox_y2',
        'detection_type', 'label', 'confidence',
        'item_bbox_x1', 'item_bbox_y1', 'item_bbox_x2', 'item_bbox_y2'
    ]

    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(csv_filepath)

    with open(csv_filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header)

        # Process each pedestrian's detections
        for ped_id, data in detections.items():
            ped_bbox = data['bbox']
            ped_x1, ped_y1, ped_x2, ped_y2 = ped_bbox

            # --- Log Apparel Detections (YOLO results) ---
            # Corrected logic to iterate over the tuple of results
            # The apparel_results variable is a tuple, so we need to iterate over each item
            for res in data['apparel_results']:
                # Ensure the item is a valid result object
                if hasattr(res, 'names') and hasattr(res, 'boxes'):
                    class_names = res.names
                    for box in res.boxes:
                        label = class_names[int(box.cls)]
                        confidence = float(box.conf)
                        # item_bbox is relative to the cropped pedestrian image
                        item_bbox_rel = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Convert item bbox to original image coordinates
                        ix1, iy1, ix2, iy2 = item_bbox_rel
                        writer.writerow([
                            frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
                            'apparel', label, f"{confidence:.4f}",
                            ped_x1 + ix1, ped_y1 + iy1, ped_x1 + ix2, ped_y1 + iy2
                        ])

            # --- Log Age and Gender Detections (List of dictionaries) ---
            age_gender_results_list = data['age_gender_results']
            
            for face_detection in age_gender_results_list:
                # Log gender
                gender_label = face_detection.get('gender')
                writer.writerow([
                    frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
                    'age_gender', gender_label, '',
                    ped_x1, ped_y1, ped_x2, ped_y2 # Use pedestrian bbox for this entry
                ])
                
                # Log age
                age_label = face_detection.get('age')
                writer.writerow([
                    frame_id, ped_id, ped_x1, ped_y1, ped_x2, ped_y2,
                    'age_gender', age_label, '',
                    ped_x1, ped_y1, ped_x2, ped_y2 # Use pedestrian bbox for this entry
                ])