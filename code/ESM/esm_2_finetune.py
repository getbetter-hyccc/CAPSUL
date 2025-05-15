import torch
import esm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import random
import numpy as np
import time
import os

# To run this program, there should be a folder named "dataset" in the current directory, which contains:
# - union.csv: a CSV file containing localization annotations.

class ProteinDataset(Dataset):
    def __init__(self, csv_file, split, max_len=4096):
        self.data = []
        self.labels = []

        with open(csv_file, "r") as file:
            lines = file.readlines()[1:]
            for line in lines:
                columns = line.strip().split(",")
                protein_id = columns[0]
                labels = list(map(int, columns[1:21])) 
                labels = [1 if label == 2 else label for label in labels] 
                # If you want to use only experimental labels, use:
                # labels = [0 if label == 2 else label for label in labels] 
                row_split = columns[21]
                sequence = columns[22][:max_len]

                if row_split == split:
                    self.data.append((protein_id, sequence))
                    self.labels.append(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_id, sequence = self.data[idx]
        labels = self.labels[idx]
        return protein_id, sequence, torch.tensor(labels, dtype=torch.float32)


class ESMMLPClassifier(nn.Module):
    def __init__(self, esm_model, input_dim=1280, hidden_dims=[512, 256, 128], output_dim=20, dropout=0.2):
        super(ESMMLPClassifier, self).__init__()
        self.esm_model = esm_model

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens):
        results = self.esm_model(tokens, repr_layers=[33])
        token_representations = results["representations"][33]

        batch_lens = (tokens != alphabet.padding_idx).sum(1)
        sequence_representations = torch.stack([
            token_representations[i, 1:tokens_len-1].mean(0)
            for i, tokens_len in enumerate(batch_lens)
        ])

        logits = self.mlp(sequence_representations)
        return logits


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  


if __name__ == "__main__":

    version = '4'
    gpu = 5
    EPOCHS = 200
    lr = 1e-4
    hidden_dims = [512, 256]
    patience = 5
    batch_size = 1

    set_seed(42)

    # Load ESM-2 model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device(f"cuda:{gpu}")
    else:
        print("Using CPU or a single GPU.")
        device = torch.device("cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    csv_file = "dataset/union.csv"

    train_dataset = ProteinDataset(csv_file, split="train")
    valid_dataset = ProteinDataset(csv_file, split="valid")
    test_dataset = ProteinDataset(csv_file, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = ESMMLPClassifier(model, input_dim=1280, hidden_dims=hidden_dims, output_dim=20, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5)

    best_valid_loss = float("inf")
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f'\nstart epoch{epoch+1}')
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            protein_ids, sequences, labels = batch
            _, _, tokens = batch_converter(list(zip(protein_ids, sequences)))
            tokens = tokens.to(device)
            labels = labels.float().to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_loss:.4f}")

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                protein_ids, sequences, labels = batch
                _, _, tokens = batch_converter(list(zip(protein_ids, sequences)))
                tokens = tokens.to(device)
                labels = labels.float().to(device)

                logits = model(tokens)
                loss = criterion(logits, labels)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_dataloader)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), f"esm_2/best_model/650m_finetune_{version}.pt")
            print("Best model saved!")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}], Val Loss: {avg_valid_loss:.4f}, Time: {elapsed_time:.2f}s")
        early_stopping(avg_valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break 

    model.load_state_dict(torch.load(f"esm_2/best_model/650m_finetune_{version}.pt", weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            protein_ids, sequences, labels = batch
            _, _, tokens = batch_converter(list(zip(protein_ids, sequences)))
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            probabilities = torch.sigmoid(logits)
            all_preds.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))

    results = {
        "y_true": all_labels,
        "y_pred": (np.array(all_preds) > 0.5).astype(int)
    }
    results_path = f"esm_2/best_model/650m_finetune_{version}_result.pt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")

    print("Classification Report on Test Set:")
    print(classification_report(all_labels, (np.array(all_preds) > 0.5).astype(int), target_names=['Nucleus', 'Nuclear Membrane', 'Nucleoli', 'Nucleoplasm', 'Cytoplasm', 'Cytosol', 'Cytoskeleton', 'Centrosome', 'Mitochondria', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Plasma Membrane/Cell Membrane', 'Endosome', 'Lipid droplet', 'Lysosome/Vacuole', 'Peroxisome', 'Vesicle', 'Primary Cilium', 'Secreted Proteins', 'Sperm'], digits = 3))
