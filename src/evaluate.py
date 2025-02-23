# evaluate.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import ISLDataset, collate_fn
from model import SignLanguageLSTM
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, data_loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, lengths, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, output_dict=True)
    return acc, report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to preprocessed npz data file")
    parser.add_argument('--model_path', type=str, default="best_model.pth", help="Path to the saved model checkpoint")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    label_map = data['label_map'].item()
    num_classes = len(label_map)
    
    test_dataset = ISLDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    acc, report = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    for label, metrics in report.items():
        print(f"{label}: {metrics}")

if __name__ == '__main__':
    main()
