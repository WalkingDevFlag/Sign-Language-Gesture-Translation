# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import SignLanguageLSTM

class ISLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # list of numpy arrays (variable-length sequences)
        self.y = y  # list of labels
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return sample, label

def collate_fn(batch):
    # Sort the batch by sequence length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    # Pad sequences to the maximum length in the batch
    padded_sequences = nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.float) for seq in sequences], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels

def train_model(train_loader, val_loader, num_classes, device, num_epochs=50, patience=5):
    model = SignLanguageLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for inputs, lengths, labels in train_bar:
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, lengths, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model for ISL gesture translation")
    parser.add_argument('--data_path', type=str, required=True, help="Path to preprocessed npz data file")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    label_map = data['label_map'].item()
    num_classes = len(label_map)
    
    train_dataset = ISLDataset(X_train, y_train)
    val_dataset = ISLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(train_loader, val_loader, num_classes, device, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
