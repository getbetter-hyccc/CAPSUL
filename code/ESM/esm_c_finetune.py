import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinTensor
from esm.tokenization import EsmSequenceTokenizer
from esm.utils.sampling import _BatchedESMProteinTensor
from Bio import SeqIO
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import random
import argparse
import time
import pandas as pd
import os

# To run this program, there should be a folder named "dataset" in the current directory, which contains:
# - union.csv: a CSV file containing localization annotations.

# - The model path should be the one you downloaded from the ESM-C website. The model we use is ESM-C 600M.


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESMC + MLP for binary classification")
    parser.add_argument('--bs', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--version', type=str, default="1", help="Model version for saving")
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[512,256,128], help="List of hidden layer sizes for MLP, e.g. --mlp_hidden_dims 512 256")
    parser.add_argument('--seed', type=int, default=42, help="Set random seed to initialize")
    parser.add_argument('--gpu', type=int, default=0, help="GPU id to use (e.g., 0, 1, 2, ...)")

    return parser.parse_args()


class ESMCWithMLP(nn.Module):
    def __init__(self, esm_model, embedding_dim=1152, hidden_dims=[512], dropout=0.2):
        super().__init__()
        self.esm = esm_model
        layers = []
        prev_dim = embedding_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 20))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batched_tensor):
        output = self.esm.logits(batched_tensor, LogitsConfig(sequence=False, return_embeddings=True))
        embedding = output.embeddings  # shape: [B, D]
        logits = self.mlp(embedding)
        return logits  # [B]
    

def read_csv(file_path, split):
    data = pd.read_csv(file_path)
    data = data[data['Dataset'] == split]
    ids = data['ID'].tolist()
    sequences = data['Sequence'].tolist()
    labels = data.iloc[:, 1:21].values
    labels[labels == 2] = 1
    # If you want to use only experimental data, use:
    # labels[labels == 2] = 0
    return ids, sequences, labels


class SequenceLabelDataset():
    def __init__(self, csv_file, split, tokenizer):
        self.ids, self.sequences, self.labels = read_csv(csv_file, split)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, label


def batch_protein_tensors(tensor_list: list[ESMProteinTensor]) -> _BatchedESMProteinTensor:
    # In order to batch the protein sequences for ESM
    def stack_field(name):
        fields = [getattr(t, name) for t in tensor_list]
        if any(f is None for f in fields):
            return None
        return torch.stack(fields, dim=0)

    return _BatchedESMProteinTensor(
        sequence=stack_field("sequence"),
        structure=stack_field("structure"),
        secondary_structure=stack_field("secondary_structure"),
        sasa=stack_field("sasa"),
        function=stack_field("function"),
        residue_annotations=stack_field("residue_annotations"),
        coordinates=stack_field("coordinates"),
    )


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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


def collate_fn(batch, model):
    # Attention Padding
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.stack(labels)
    
    tokenized = model.esm._tokenize(sequences)  # tokenized tensor list
    tensor_list = [ESMProteinTensor(sequence=x) for x in tokenized]
    batched_tensor = batch_protein_tensors(tensor_list)
    
    return batched_tensor, torch.tensor(labels, dtype=torch.float32)


def train_esmc_with_mlp(csv_file,
                        model, device, batch_size=4, lr=5e-5,
                        epochs=100, patience=10, version="ft"):

    tokenizer = model.esm.tokenizer
    train_set = SequenceLabelDataset(csv_file, 'train', tokenizer)
    val_set = SequenceLabelDataset(csv_file, 'valid', tokenizer)
    test_set = SequenceLabelDataset(csv_file, 'test', tokenizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, model))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, model))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    best_path = f"esm_c/best_model/esmc_finetune_{version}.pt"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    print("Now start to train!")
    for epoch in range(epochs):
        start_time = time.time()
        print(f"\nStart epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                output = model(batch)
                val_loss += criterion(output, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"End epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s")

        if val_loss < early_stopping.best_loss - early_stopping.min_delta:
            torch.save(model.state_dict(), best_path)
            print(f"Saved Best Model at Epoch {epoch+1} (Val Loss: {val_loss:.4f})")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break


    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            output = torch.sigmoid(model(batch)).cpu().numpy()
            preds.extend(output)
            trues.extend(labels.numpy())

    preds = np.array(preds) > 0.5
    trues = np.array(trues)

    results = {"y_true": trues, "y_pred": preds}
    results_path = f"esm_c/best_model/esmc_finetune_{version}_results.pt"
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")

    print("\nTest Set Classification Report:")
    print(classification_report(trues, preds, target_names=['Nucleus', 'Nuclear Membrane', 'Nucleoli', 'Nucleoplasm', 'Cytoplasm', 'Cytosol', 'Cytoskeleton', 'Centrosome', 'Mitochondria', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Plasma Membrane/Cell Membrane', 'Endosome', 'Lipid droplet', 'Lysosome/Vacuole', 'Peroxisome', 'Vesicle', 'Primary Cilium', 'Secreted Proteins', 'Sperm'], digits = 3))



def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_seed(args.seed)
    

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # The model path should be the one you downloaded from the ESM-C website
    model_path = "esm_c/model/esmc_600m_2024_12_v0.pth"
    tokenizer = EsmSequenceTokenizer()
    base_model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=True
    )
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model = base_model.to(device)
    full_model = ESMCWithMLP(esm_model=base_model, embedding_dim=1152, hidden_dims=args.mlp_hidden_dims).to(device)
    csv_file = 'dataset/union.csv'
    
    train_esmc_with_mlp(
        csv_file = csv_file,
        model=full_model,
        device=device,
        batch_size=args.bs,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        version=args.version
    )

