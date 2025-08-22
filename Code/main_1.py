# main.py

# Import necessary libraries
from collections import Counter
import cv2
import csv
import datetime
import joblib
import math
import numpy as np
import os
import pandas as pd
from models_infer_1 import pedestrian_crop, detect_pedestrian, log_detections_to_csv
import warnings

# --- Configuration for testing ---
ARTIFACT_PATH = "../Models/ml_model/recommender_artifacts_singular.joblib"
# The model path is now handled by the artifacts file
TOP_K = 5

def run_recommender(csv_file_path):
    """
    Takes the CSV log as input, processes the data, and outputs ad recommendations
    based on the loaded machine learning model.
    """
    print(f"\n--- Running recommender model on: {csv_file_path} ---")

    # 1. Load the saved artifacts
    try:
        artifacts = joblib.load(ARTIFACT_PATH)
        model = artifacts["model"]
        age_scaler = artifacts["age_scaler"]
        gender_enc = artifacts["gender_enc"]
        label_enc = artifacts["label_enc"]
        apparel_cols = artifacts["apparel_cols"]
        feature_columns = artifacts["feature_columns"]
        print("Recommender artifacts loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The artifact file '{ARTIFACT_PATH}' was not found. Please ensure it has been created by the training script.")
        return
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # 2. Process the CSV log
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The log file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    if df.empty:
        print("CSV file is empty. No recommendations to generate.")
        return

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(csv_file_path)), 'Outputs')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    recommender_output_path = os.path.join(output_dir, f"recommender_log_{timestamp}.csv")
    
    # Define the header for the recommender CSV
    header = ['pedestrian_id', 'rank', 'ad_topic', 'confidence']
    
    # Aggregate data by pedestrian_id to create a "user profile"
    user_profiles = {}
    for pedestrian_id in df['pedestrian_id'].unique():
        pedestrian_data = df[df['pedestrian_id'] == pedestrian_id]
        
        # Get age and gender
        age_gender_data = pedestrian_data[pedestrian_data['detection_type'] == 'age_gender']
        if not age_gender_data.empty:
            age = age_gender_data[age_gender_data['label'].str.contains(r'^(\d+)-(\d+)$', regex=True)]['label'].iloc[0] if not age_gender_data[age_gender_data['label'].str.contains(r'^(\d+)-(\d+)$', regex=True)].empty else 'unknown'
            gender = age_gender_data[age_gender_data['label'].isin(['Male', 'Female'])]['label'].iloc[0] if not age_gender_data[age_gender_data['label'].isin(['Male', 'Female'])].empty else 'unknown'
        else:
            age = 'unknown'
            gender = 'unknown'

        # Get apparel
        apparel_data = pedestrian_data[pedestrian_data['detection_type'] == 'apparel']
        apparel_list = apparel_data['label'].tolist()
        
        # Store the profile
        user_profiles[pedestrian_id] = {
            'Age': age,
            'Gender': gender,
            'Apparel': apparel_list
        }

    # 3. Make a prediction for each pedestrian and log the results
    with open(recommender_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for ped_id, profile in user_profiles.items():
            print(f"\n--- Processing Pedestrian ID: {ped_id} ---")
            
            user_age_str = profile['Age']
            # Handle age ranges by taking the lower bound
            if user_age_str != 'unknown':
                user_age = int(user_age_str.split('-')[0])
            else:
                user_age = 30 # A reasonable default for unknown age
                
            user_gender = profile['Gender']
            user_apparel = profile['Apparel']

            # Preprocess the user profile data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

                user_df = pd.DataFrame({
                    'Age': [user_age],
                    'Gender': ['Male' if user_gender == 'unknown' else user_gender]
                })

                user_df['age_norm'] = age_scaler.transform(user_df[['Age']].astype(float))
                user_df['gender_enc'] = gender_enc.transform(user_df['Gender'].astype(str))

                apparel_features = {col: 0 for col in apparel_cols}
                for item in user_apparel:
                    if f'apparel_{item}' in apparel_features:
                        apparel_features[f'apparel_{item}'] = 1

                apparel_df = pd.DataFrame([apparel_features])

                def block_normalize(block_df):
                    if block_df.shape[1] == 0:
                        return block_df
                    return block_df.divide(math.sqrt(block_df.shape[1]))

                apparel_block = block_normalize(apparel_df)

                user_features = pd.concat([user_df[['age_norm', 'gender_enc']].reset_index(drop=True), apparel_block.reset_index(drop=True)], axis=1)
                user_features = user_features.reindex(columns=feature_columns, fill_value=0)

            # Make a prediction
            probabilities = model.predict_proba(user_features)[0]

            # Get and display the top K recommendations
            top_k_indices = np.argsort(probabilities)[::-1][:TOP_K]

            print(f"\nTop {TOP_K} Recommended Ad Topics:")
            for i, index in enumerate(top_k_indices):
                ad_topic = label_enc.inverse_transform([index])[0]
                confidence = probabilities[index]
                print(f"  {i+1}. {ad_topic}: {confidence:.4f} confidence")

                # Log the recommendation to the CSV file
                writer.writerow([ped_id, i+1, ad_topic, f"{confidence:.4f}"])

    print(f"\n--- Recommender model finished. Results logged to {recommender_output_path} ---")


# Placeholder for machine learning recommender model
def classify(csv_file_path):
    """
    Placeholder function for a machine learning model that processes
    the detection data from a CSV file. This is no longer used,
    as run_recommender is now the primary function.
    """
    print("Placeholder: The 'classify' function has been replaced with 'run_recommender'.")
    pass

def main():
    print("Starting the Computer Vision Project...")

    # Generate a timestamp for the output filename for better trackability
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define paths
    image_path = r"C:\Users\Rajarshi\Pictures\capstone_test_images\test1.jpg"
    csv_output_path = f'../../logs/detections/detection_log_{timestamp}.csv' # Corrected path to be relative

    # Check if the logs directory exists, create if not
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    # Check if the example image exists, if not, create a dummy one.
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Creating a dummy black image for demonstration.")
        dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_image)
        print(f"Dummy image created at '{image_path}'. Please replace it with a real image.")

    # 1. Load an image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Run pedestrian detection
    print("Detecting pedestrians...")
    # NOTE: Assuming detect_pedestrian is defined and available
    pedestrian_results = detect_pedestrian(image)
    
    # Check if any pedestrians were detected
    if not any(r.boxes for r in pedestrian_results):
        print("No pedestrians detected in the image.")
    else:
        # 3. Crop pedestrians and detect apparel, age, and gender
        print("Cropping pedestrians and detecting attributes...")
        all_detections = pedestrian_crop(pedestrian_results, image)

        # 4. Log all detections to a CSV file
        print(f"Logging detections to {csv_output_path}...")
        log_detections_to_csv(all_detections, csv_output_path, frame_id=0)
        print("Logging complete.")

        # 5. Feed the CSV to the recommender model
        run_recommender(csv_output_path)
        
    print("\nProject pipeline finished.")

if __name__ == "__main__":
    main()


