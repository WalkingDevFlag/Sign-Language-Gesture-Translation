# preprocessing.py
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_keypoints_from_video(video_path):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Extract keypoints from pose and both hands
        frame_keypoints = []
        # Pose: 33 landmarks × 4 (x,y,z,visibility)
        if results.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                             for landmark in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*4)
        frame_keypoints.extend(pose)

        # Left hand: 21 landmarks × 3 (x,y,z)
        if results.left_hand_landmarks:
            left_hand = np.array([[landmark.x, landmark.y, landmark.z] 
                                  for landmark in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(21*3)
        frame_keypoints.extend(left_hand)

        # Right hand: 21 landmarks × 3 (x,y,z)
        if results.right_hand_landmarks:
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] 
                                   for landmark in results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand = np.zeros(21*3)
        frame_keypoints.extend(right_hand)

        keypoints_list.append(np.array(frame_keypoints))
    cap.release()
    holistic.close()
    return np.array(keypoints_list)

def augment_keypoints(keypoints, noise_factor=0.01):
    # Simple augmentation: add Gaussian noise to keypoints
    noise = np.random.normal(0, noise_factor, keypoints.shape)
    return keypoints + noise

def process_dataset(dataset_path, test_size=0.2, augment=False):
    X = []
    y = []
    label_map = {}
    label_idx = 0
    # Iterate through each gesture folder
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            if folder not in label_map:
                label_map[folder] = label_idx
                label_idx += 1
            # Use lower() to ensure case-insensitive matching for file extensions
            video_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if not video_files:
                print(f"Warning: No video files found in folder {folder_path}")
                continue
            for video_file in tqdm(video_files, desc=f"Processing {folder}"):
                video_path = os.path.join(folder_path, video_file)
                try:
                    keypoints = extract_keypoints_from_video(video_path)
                    # Optionally apply augmentation
                    if augment:
                        keypoints = augment_keypoints(keypoints)
                    X.append(keypoints)
                    y.append(label_map[folder])
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    if len(X) == 0:
        raise ValueError("No video samples were found. Please check the dataset path, folder structure, and file extensions.")
    # Split dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    return (X_train, y_train), (X_test, y_test), label_map

def save_dataset(data, output_path):
    np.savez_compressed(output_path, **data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess ISL videos to extract keypoints")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the preprocessed data (npz file)")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test), label_map = process_dataset(args.dataset_path, augment=args.augment)
    
    data = {
        'X_train': np.array(X_train, dtype=object),  # variable-length sequences
        'y_train': np.array(y_train),
        'X_test': np.array(X_test, dtype=object),
        'y_test': np.array(y_test),
        'label_map': label_map
    }
    save_dataset(data, args.output_path)
    print(f"Preprocessed data saved to {args.output_path}")
