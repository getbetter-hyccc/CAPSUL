import torch
from torchdrug import transforms, datasets, data, utils
import os
import glob
import pandas as pd
import random
import numpy as np
import copy
from sklearn.metrics import classification_report
from torch.utils import data as torch_data
from torchdrug.utils import comm, pretty
from itertools import islice
from torch import nn
from tqdm import tqdm
import warnings
from rdkit import Chem
import logging
logger = logging.getLogger(__name__)
from torchdrug import layers, tasks
from torchdrug.layers import geometry
from torchdrug import models
from torch.nn import functional as F
from torchdrug import core
from collections.abc import Sequence
from torch_scatter import scatter_add
from torchdrug.layers.readout import Readout

# To run this program, there should be a folder named "dataset" in the current directory, which contains:
# - pdb/AlphaFold_pdb.tar: the selected .pdb original file.
# - union.csv: a CSV file containing localization annotations.

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='GearNet-Edge+Transformer')
    parser.add_argument('--version', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--gpus', default=[0], type=list)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--hidden_dims', default=[512,512,512], type=list)
    parser.add_argument('--concat_hidden', default=True, type=bool, help='Whether to concatenate hidden states')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')

    parser.add_argument('--transformer_input_dim', default=1536, type=int, help='Input dimension of transformer')
    parser.add_argument('--transformer_hidden_dim', default=1536, type=int, help='Hidden dimension of transformer')
    parser.add_argument('--transformer_num_heads', default=2, type=int, help='Number of heads in transformer')
    parser.add_argument('--transformer_num_layers', default=2, type=int, help='Number of layers in transformer')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--early_stopping_patience', default=8, type=int)


    args = parser.parse_args()
    return args

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NewAlphaFold(datasets.AlphaFoldDB):
    def __init__(self, label_file, verbose=1, gpu=0, **kwargs):
        self.gpu = gpu
        splits = ["train", "valid", "test"]
        self.label_df = pd.read_csv(label_file)  
        self.label_dict = {
            row["ID"]: [1 if x == 2 else x for x in row.iloc[1:21].tolist()]  
            for _, row in self.label_df.iterrows()
        }

        # If you want to use only the experimental data, use:
        # row["ID"]: [0 if x == 2 else x for x in row.iloc[1:21].tolist()]  

        self.split_dict = {
            row["ID"]: row.iloc[21]
            for _, row in self.label_df.iterrows()
        }
        self.data_dict = {}
        self.pdb_files_dict = {}
        
        exist = True
        for split in splits:
            pkl_file = f"dataset/pdb/{split}/{split}.pkl.gz"
            if not os.path.exists(pkl_file):
                exist = False
                break
        
        # If you run for the first time, it would take you some time to construct .pkl file from .tar file
        if exist == True:
            for split in splits:
                pkl_file = f"dataset/pdb/{split}/{split}.pkl.gz"
                self.load_pickle(pkl_file, verbose=verbose, **kwargs)
                pdb_temp = self.pdb_files
                data_temp = self.data
                self.pdb_files_dict[split] = copy.deepcopy(pdb_temp)
                self.data_dict[split] = copy.deepcopy(data_temp)
                self.pdb_files = []
                self.data = []
                print(f"Loaded {split} data from {pkl_file}")
        else:
            tar_file = "dataset/pdb/Alphafold_pdb.tar" 
            pdb_path = utils.extract(tar_file)
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"Extracted path {pdb_path} does not exist. Check the tar file content.")

            pdb_files_list = sorted(glob.glob(os.path.join(pdb_path, "*.pdb")))
            if not pdb_files_list:
                raise ValueError(f"No PDB files found in extracted path {pdb_path}")


            split_pdb_files = {"train": [], "valid": [], "test": []}
            for pdb_file in pdb_files_list:
                filename = os.path.basename(pdb_file)
                protein_id = filename.split('-')[0] 
                split = self.split_dict.get(protein_id)
                if split in split_pdb_files:
                    split_pdb_files[split].append(pdb_file)

            for split, pdb_files in split_pdb_files.items():
                if not pdb_files:
                    print(f"No PDB files found for split '{split}'. Skipping...")
                    continue

                print(f"Processing {len(pdb_files)} PDB files for split '{split}'...")
                self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
                pdb_temp = self.pdb_files
                data_temp = self.data
                self.pdb_files_dict[split] = copy.deepcopy(pdb_temp)
                self.data_dict[split] = copy.deepcopy(data_temp)

                pkl_file = f"dataset/pdb/{split}.pkl.gz"
                self.save_pickle(pkl_file, verbose=verbose)
                print(f"Saved {split} data to {pkl_file}")

                self.pdb_files = []
                self.data = []
        
        
        tasks = ['Nucleus', 'Nuclear Membrane', 'Nucleoli', 'Nucleoplasm', 'Cytoplasm', 'Cytosol', 'Cytoskeleton', 'Centrosome', 'Mitochondria', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Plasma Membrane/Cell Membrane', 'Endosome', 'Lipid droplet', 'Lysosome/Vacuole', 'Peroxisome', 'Vesicle', 'Primary Cilium', 'Secreted Proteins', 'Sperm']
        task2id = {task: i for i, task in enumerate(tasks)}
        self.targets = task2id


    @utils.copy_args(data.Protein.from_molecule)
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)
    



    def get_item(self, index, split):

        if split not in self.data_dict:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {list(self.data.keys())}")

        if index >= len(self.pdb_files_dict[split]):
            raise IndexError(f"Index {index} out of range for split '{split}' with {len(self.pdb_files_dict[split])} samples")
        
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files_dict[split][index], self.kwargs)
        else:
            protein = self.data_dict[split][index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        

        filename = os.path.basename(self.pdb_files_dict[split][index])
        protein_id = filename.split('-')[0]
        label = self.label_dict.get(protein_id, [0] * 20)
        
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        
        
        item["targets"] = torch.tensor(label, dtype=torch.float32).to(torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu'))
        return item


class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
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
          # attn_weights shape: (batch_size, seq_len, seq_len)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights
        else:
            return src


class NewMultipleBinaryClassification(tasks.MultipleBinaryClassification):
    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), normalization=True, reweight=False, graph_construction_model=None, verbose=0, input_dim=1536, hidden_dim=1536, num_heads=16, num_layers=10, num_classes=1):

        super(tasks.MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        encoder_layer = TransformerEncoderLayerWithAttention(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric, return_attention = False)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric
    
    def predict(self, batch, all_loss=None, metric=None, return_attention=False):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.input_dim != self.hidden_dim:
            hidden_features = self.input_fc(output["graph_feature"])
        else:
            hidden_features = output["graph_feature"]

        # attention padding
        padding_mask = output["attention_mask"].bool()
        if return_attention:
            attention_matrices = []
            x = hidden_features
            for layer in self.transformer.layers:
                x, attn_weights = layer(src=x, src_key_padding_mask=~padding_mask, return_attention=True)
                attention_matrices.append(attn_weights)
            transformer_output = x  
        else:
            transformer_output = self.transformer(hidden_features, src_key_padding_mask=~padding_mask)
        valid_positions = padding_mask.sum(dim=1, keepdim=True)
        graph_feature = transformer_output.sum(dim=1) / valid_positions.float()
        pred = self.fc(graph_feature)

        if return_attention:
            return pred, torch.stack(attention_matrices)
        else:
            return pred


class CustomEngine(core.Engine):
    def train(self, num_epoch=1, batch_per_epoch=None, early_stopping_patience=None, version=0):
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                
                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

            _, eval_loss, _, _, _ = self.evaluate("valid", log=True, return_attention=False)
            print(f"loss on validation dataset: {eval_loss}")
            
            # Save best model and handle early stopping
            if eval_loss < best_loss:
                best_loss = eval_loss
                patience_counter = 0
                if self.rank == 0:  # Only save from rank 0 process
                    torch.save(model.state_dict(), f"torchdrug+T/best_model/torchdrug+T_{version}.pth")
                    print(f"Saved best model with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    @torch.no_grad()
    def evaluate(self, split, log=True, return_attention=False):
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        losses = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
            if return_attention:
                pred, attention= model.predict(batch, return_attention=return_attention)
            else:
                pred = model.predict(batch, return_attention=return_attention)
            target = model.target(batch)
            loss, _ = model(batch)
            preds.append(pred)
            targets.append(target)
            losses.append(loss.item())

        pred = utils.cat(preds)
        target = utils.cat(targets)
        avg_loss = sum(losses) / len(losses)

        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        if return_attention:
            return metric, avg_loss, pred, target, attention
        else:
            return metric, avg_loss, pred, target, None


class KeepReadout(Readout):
    # A necessary readout replacement, as we replace the MLP with a transformer
    def forward(self, graph, input):
        input2graph = self.get_index2graph(graph)
        unique_elements = torch.unique(input2graph)
        graph_num = unique_elements.size(0)
        output = []
        for graph_idx in range(graph_num):
            graph_nodes_indices = [i for i, g in enumerate(input2graph) if g == graph_idx]
            graph_nodes = input[graph_nodes_indices]
            output.append(graph_nodes)

        max_len = max(t.shape[0] for t in output)
        padded_output = []
        attention_masks = []
        for tensor in output:
            pad_size = max_len - tensor.shape[0]
            padded_tensor = torch.cat([tensor, torch.zeros(pad_size, *tensor.shape[1:]).to(tensor.device)], dim=0)
            padded_output.append(padded_tensor)

            attention_mask = torch.cat([torch.ones(tensor.shape[0]), torch.zeros(pad_size)]).to(tensor.device)
            attention_masks.append(attention_mask)
        
        final_output = torch.stack(padded_output)
        final_attention_mask = torch.stack(attention_masks)
        return final_output, final_attention_mask


class CustomGearNet(models.GearNet):
    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None, short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(models.GearNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "keep":
            self.readout = KeepReadout()
            print("Now you are keeping the output without pooling!")
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        
    def forward(self, graph, input):
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        graph_feature, attention_mask = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "attention_mask": attention_mask
        }


if __name__ == '__main__':
    set_seed(42)
    args = parse_args()

    # preprocessing of data
    truncate_transform = transforms.TruncateProtein(max_length=3000, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    # load the dataset
    dataset = NewAlphaFold(label_file="dataset/union.csv", transform=transform, atom_feature=None, bond_feature=None, gpu = args.gpu)
    train_set = [dataset.get_item(i, "train") for i in range(len(dataset.pdb_files_dict['train']))]
    valid_set = [dataset.get_item(i, "valid") for i in range(len(dataset.pdb_files_dict['valid']))]
    test_set = [dataset.get_item(i, "test") for i in range(len(dataset.pdb_files_dict['test']))]

    gearnet = CustomGearNet(input_dim=21, hidden_dims=args.hidden_dims, num_relation=7, edge_input_dim=59, num_angle_bin=8, batch_norm=True, concat_hidden=args.concat_hidden, short_cut=True, readout="keep")
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                    geometry.KNNEdge(k=10, min_distance=5),
                                                                    geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")
    task = NewMultipleBinaryClassification(model=gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=args.num_mlp_layer, task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["f1_max"], input_dim=args.transformer_input_dim, hidden_dim=args.transformer_hidden_dim, num_heads=args.transformer_num_heads, num_layers=args.transformer_num_layers, num_classes=20)
    optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)
    solver = CustomEngine(task, train_set, valid_set, test_set, optimizer, gpus=args.gpus, batch_size=args.batch_size)
    solver.train(num_epoch=args.epoch, early_stopping_patience=args.early_stopping_patience, version = args.version)

    pred = []
    target = []
    _, _, pred, target, attention = solver.evaluate("test", return_attention=True)
    pred = torch.sigmoid(pred).cpu().numpy()
    target = target.cpu().numpy()
    threshold = 0.5
    pred_labels = (pred >= threshold).astype(int)
    target_labels = target.astype(int)

    results = {"y_true": target_labels, "y_pred": pred_labels}
    results_path = f"torchdrug+T/best_model/torchdrug+T_{args.version}_result.pt"
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")

    print("\nTest Set Classification Report:")
    print(classification_report(target_labels, pred_labels, target_names=['Nucleus', 'Nuclear Membrane', 'Nucleoli', 'Nucleoplasm', 'Cytoplasm', 'Cytosol', 'Cytoskeleton', 'Centrosome', 'Mitochondria', 'Endoplasmic Reticulum', 'Golgi Apparatus', 'Plasma Membrane/Cell Membrane', 'Endosome', 'Lipid droplet', 'Lysosome/Vacuole', 'Peroxisome', 'Vesicle', 'Primary Cilium', 'Secreted Proteins', 'Sperm'], digits = 3))