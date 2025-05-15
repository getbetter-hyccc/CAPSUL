import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils import orientation
import pandas as pd

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i

class LocalizationDataset(Dataset):

    def __init__(self, root='/tmp/protein-data/fold', random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        npy_dir = os.path.join(root, 'coordinate')
        fasta_file = os.path.join(root, 'union.csv')

        # Load the fasta file.
        protein_seqs = []
        csv_data = pd.read_csv(fasta_file)
        filtered_data = csv_data[csv_data['Dataset'] == self.split]

        for _, row in filtered_data.iterrows():
            protein_name = row['ID']
            amino_chain = row['Fasta Sequence']
            amino_ids = []
            for amino in amino_chain:
                amino_ids.append(aa_to_id[amino])
            protein_seqs.append((protein_name, np.array(amino_ids)))

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+"-1.npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            # Load labels from the CSV file
            label_data = csv_data[csv_data['ID'] == protein_name].iloc[:, 1:21].values.flatten()
            label_data[label_data == 2] = 1
            # If you want to use only the experimental subcellular localization annotations, use:
            # label_data[label_data == 2] = 0
            self.labels.append(torch.tensor(label_data.astype(int)))

        self.num_classes = 20


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]
        return data