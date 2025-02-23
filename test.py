# test.py
import argparse
import cv2
import torch
import numpy as np
from inference import infer, extract_keypoints_from_video
from model import SignLanguageLSTM

def test_inference(video_path, model_path, label_map_path):
    data = np.load(label_map_path, allow_pickle=True)
    label_map = data['label_map'].item()
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    predicted = infer(video_path, model, device, label_map)
    print(f"Predicted Gesture for video {video_path}: {predicted}")

def test_live_inference(model_path, label_map_path):
    # Live inference using webcam
    data = np.load(label_map_path, allow_pickle=True)
    label_map = data['label_map'].item()
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    print("Starting live inference. Press 'q' to quit.")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Live Feed", frame)
        # Process frame for keypoints (same extraction as in inference)
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
        
        frames.append(np.array(frame_keypoints))
        
        # Run inference every 30 frames
        if len(frames) == 30:
            input_seq = torch.tensor(frames, dtype=torch.float).unsqueeze(0).to(device)
            lengths = torch.tensor([len(frames)], dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                output = model(input_seq, lengths)
                _, pred = torch.max(output, 1)
            inv_label_map = {v: k for k, v in label_map.items()}
            print(f"Predicted Gesture: {inv_label_map[pred.item()]}")
            frames = []
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Test script for ISL gesture translation components")
    parser.add_argument('--video_path', type=str, help="Path to video file for batch inference")
    parser.add_argument('--live', action='store_true', help="Enable live inference from webcam")
    parser.add_argument('--model_path', type=str, default="best_model.pth", help="Path to saved model checkpoint")
    parser.add_argument('--label_map_path', type=str, required=True, help="Path to npz file containing label_map")
    args = parser.parse_args()
    
    if args.live:
        test_live_inference(args.model_path, args.label_map_path)
    elif args.video_path:
        test_inference(args.video_path, args.model_path, args.label_map_path)
    else:
        print("Please provide either --video_path for batch inference or --live for live inference.")

if __name__ == '__main__':
    main()
