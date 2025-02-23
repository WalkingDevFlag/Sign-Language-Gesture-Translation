# inference.py
import cv2
import mediapipe as mp
import numpy as np
import torch
from model import SignLanguageLSTM
from tqdm import tqdm
import argparse

def extract_keypoints_from_video(video_path):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame_keypoints = []
        if results.pose_landmarks:
            pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                             for landmark in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*4)
        frame_keypoints.extend(pose)

        if results.left_hand_landmarks:
            left_hand = np.array([[landmark.x, landmark.y, landmark.z]
                                  for landmark in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(21*3)
        frame_keypoints.extend(left_hand)

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

def infer(video_path, model, device, label_map):
    keypoints = extract_keypoints_from_video(video_path)
    # Convert to tensor and add batch dimension
    input_seq = torch.tensor(keypoints, dtype=torch.float).unsqueeze(0)
    length = torch.tensor([keypoints.shape[0]], dtype=torch.long)
    input_seq, length = input_seq.to(device), length.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_seq, length)
        _, pred = torch.max(output, 1)
    # Inverse mapping from label index to gesture name
    inv_label_map = {v: k for k, v in label_map.items()}
    predicted_gesture = inv_label_map[pred.item()]
    return predicted_gesture

def main():
    parser = argparse.ArgumentParser(description="Inference script for ISL gesture translation")
    parser.add_argument('--video_path', type=str, required=True, help="Path to input video")
    parser.add_argument('--model_path', type=str, default="best_model.pth", help="Path to saved model checkpoint")
    parser.add_argument('--label_map_path', type=str, required=True, help="Path to npz file containing label_map")
    args = parser.parse_args()

    # Load label_map from the provided npz file
    data = np.load(args.label_map_path, allow_pickle=True)
    label_map = data['label_map'].item()
    num_classes = len(label_map)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    predicted = infer(args.video_path, model, device, label_map)
    print(f"Predicted Gesture: {predicted}")

if __name__ == '__main__':
    main()
