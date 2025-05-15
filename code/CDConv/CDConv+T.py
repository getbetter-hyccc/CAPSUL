import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datasets import LocalizationDataset
from torch_geometric.loader import DataLoader
from models import Model, BasicBlock
from sklearn.metrics import classification_report
import random
import time

# To run this program, there should be a folder named "dataset" in the current directory, which contains:
# - coordinate: a folder containing .npy files with protein coordinates.
# - union.csv: a CSV file containing localization annotations.

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CDConv+T')
    parser.add_argument('--data-dir', default='dataset', type=str, metavar='N', help='data root directory')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=5, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', nargs='+', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=64, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512], type=int, metavar='N', help='feature channels')
    parser.add_argument('--num-epochs', default=400, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[100, 300], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--version', default=None, type=str, help='name the path where to save checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--transformer_heads', default=2, type=int, help='num of transformer heads')
    parser.add_argument('--transformer_layers', default=2, type=int, help='num of transformer layers')

    args = parser.parse_args()
    return args


class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    # Custom Transformer Encoder Layer with attention weights
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False, is_causal=False): 

        if return_attention:
            attn_output, attn_weights = self.self_attn(
                src, src, src, key_padding_mask=src_key_padding_mask, average_attn_weights=True
            )
        else:
            attn_output, _ = self.self_attn(
                src, src, src, key_padding_mask=src_key_padding_mask, average_attn_weights=True
            )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if return_attention:
            return src, attn_weights
        else:
            return src


class CustomCDConvModel(Model):
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 20,
                 num_heads: int = 2,
                 num_layers: int = 2) -> nn.Module:

        super().__init__(
            geometric_radii=geometric_radii,
            sequential_kernel_size=sequential_kernel_size,
            kernel_channels=kernel_channels,
            channels=channels
        )

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]
        self.layers = nn.Sequential(*layers)
        encoder_layer = TransformerEncoderLayerWithAttention(d_model=channels[-1], nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def forward(self, data, return_attention=False):
        x, pos, seq, ori, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.batch)
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)

        # batch info
        unique_elements = torch.unique(batch)
        batch_size = unique_elements.size(0)
        batch_output = []
        for graph_idx in range(batch_size):
            graph_nodes_indices = [i for i, g in enumerate(batch) if g == graph_idx]
            graph_nodes = x[graph_nodes_indices]
            batch_output.append(graph_nodes)

        # attention padding
        max_len = max(t.shape[0] for t in batch_output)
        padded_output = []
        attention_masks = []
        for tensor in batch_output:
            pad_size = max_len - tensor.shape[0]
            padded_tensor = torch.cat([tensor, torch.zeros(pad_size, *tensor.shape[1:]).to(tensor.device)], dim=0)
            padded_output.append(padded_tensor)

            attention_mask = torch.cat([torch.ones(tensor.shape[0]), torch.zeros(pad_size)]).to(tensor.device)
            attention_masks.append(attention_mask)
        x = torch.stack(padded_output)
        padding_mask = torch.stack(attention_masks).bool()

        if return_attention:
            attention_matrices = []
            for layer in self.transformer.layers:
                x, attn_weights = layer(src=x, src_key_padding_mask=~padding_mask, return_attention=True)
                attention_matrices.append(attn_weights)
            out = x 
        else:
            out = self.transformer(src=x,src_key_padding_mask=~padding_mask)
        
        valid_positions = padding_mask.sum(dim=1, keepdim=True)
        out = out.sum(dim=1) / valid_positions.float()
        out = self.fc(out)

        if return_attention:
            print(torch.stack(attention_matrices).shape)
            return out, torch.stack(attention_matrices) # (num_layers, batch_size, seq_len, seq_len)
            #unifinished
        else:
            return out

def train(dataloader):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        data.y = data.y.view(output.shape)  # adjust the shape to [batch_size, num_classes]
        loss = F.binary_cross_entropy_with_logits(output, data.y.float(), reduction='sum')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        total_loss += loss.item()
        total_samples += data.y.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss


def test(dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data, return_attention=False)
            data.y = data.y.view(output.shape)  # adjust the shape to [batch_size, num_classes]
            loss = F.binary_cross_entropy_with_logits(output, data.y.float(), reduction='sum')
            total_loss += loss.item()
            total_samples += data.y.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss


def set_seed(seed=42):
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    print(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_dataset = LocalizationDataset(root=args.data_dir, random_seed=args.seed, split='train')
    print("Finished construct train_dataset!")
    valid_dataset = LocalizationDataset(root=args.data_dir, random_seed=args.seed, split='valid')
    print("Finished construct valid_dataset!")
    test_dataset = LocalizationDataset(root=args.data_dir, random_seed=args.seed, split='test')
    print("Finished construct test_dataset!")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = CustomCDConvModel(geometric_radii=[2*args.geometric_radius, 4*args.geometric_radius],
                  sequential_kernel_size=args.sequential_kernel_size,
                  kernel_channels=args.kernel_channels, channels=args.channels, base_width=args.base_width,
                  num_classes=train_dataset.num_classes, num_heads=args.transformer_heads, num_layers=args.transformer_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)

    # learning rate scheduler
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i-1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)


    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float('inf')
            self.early_stop = False

        def __call__(self, val_loss):
            print(val_loss)
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopping = EarlyStopping(patience=args.patience)
    best_valid_loss = float("inf")
    best_epoch = 0
    print("Start to train")   
    model_path = f"CDConv+T/best_model/CDConv+T_{args.version}_model.pt"
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(args.num_epochs):
        print(f'Epoch: {epoch+1:03d}')
        start_time = time.time()
        train_avg_loss = train(train_loader)
        lr_scheduler.step()
        print(f'Training Average Loss: {train_avg_loss:.4f}')
        valid_avg_loss = test(valid_loader)
        print(f'Validation Average Loss: {valid_avg_loss:.4f}')
        if valid_avg_loss <= best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_avg_loss
            checkpoint = model.state_dict()
            torch.save(checkpoint, model_path)
            print("Best model saved!")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch [{epoch+1}], Time: {elapsed_time:.2f}s")
        early_stopping(valid_avg_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break 


    model.load_state_dict(torch.load(model_path))
    model.eval()
    total_loss = 0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data, return_attention=False)
            data.y = data.y.view(output.shape)
            loss = F.binary_cross_entropy_with_logits(output, data.y.float(), reduction='sum')
            total_loss += loss.item()
            total_samples += data.y.size(0)
            preds = (torch.sigmoid(output) > 0.5).int()
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # 计算平均损失
    test_avg_loss = total_loss / total_samples if total_samples > 0 else 0

    # 保存预测结果和实际结果
    results = {"y_true": y_true, "y_pred": y_pred}
    results_path = f"CDConv+T/best_model/CDConv+T_{args.version}_results.pt"
    torch.save(results, results_path)
    print(f"Test Average Loss: {test_avg_loss:.4f}")
    print(f"Results saved to {results_path}")
    
    
    # 输出分类报告
    report = classification_report(y_true, y_pred, target_names=['Nucleus', 'Nuclear Membrane', 'Nucleoli', 'Nucleoplasm', 'Cytoplasm', 'Cytosol', 'Cytoskeleton', 'Centrosome', 'Mitochondria', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Plasma Membrane/Cell Membrane', 'Endosome', 'Lipid droplet', 'Lysosome/Vacuole', 'Peroxisome', 'Vesicle', 'Primary Cilium', 'Secreted Proteins', 'Sperm'], digits=3)
    print(report)
