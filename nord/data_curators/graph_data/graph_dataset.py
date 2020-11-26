import os

import torch
from dgl import load_graphs, save_graphs
from dgl.data.utils import load_info, save_info
from torch.utils.data import Dataset

from nord.utils import get_lower_index


class GraphDataset(Dataset):

    @classmethod
    def __from_file__(cls, path, all_vs_all):
        ds = GraphDataset([], [], None, all_vs_all)
        ds.load(path)
        ds.graphs_number = len(ds.labels)
        return ds

    def __init__(self, graphs, labels, classes, all_vs_all=True):
        """
        Args

        """
        self.graphs = graphs
        self.labels = labels
        self.classes = classes

        self.graphs_number = len(self.labels)
        self.e = 0.003  # e% difference

        self.all_vs_all = all_vs_all

    def __len__(self):
        if self.classes == 1:
            return self.graphs_number

        if not self.all_vs_all:
            return self.graphs_number*(self.graphs_number+1)//2

        return self.graphs_number**2

    def __getitem__(self, idx):

        id_1 = idx % self.graphs_number
        id_2 = idx // self.graphs_number

        if not self.all_vs_all:
            # Sample only the lower triangle+diagonal
            id_1, id_2 = get_lower_index(idx)
            # k = np.floor(-0.5 + (np.sqrt(0.25+2*idx)))
            # j = idx - k*(k+1)/2
            # id_1 = int(k)
            # id_2 = int(j)

        label = None

        if self.classes == 3:

            label = [0, 1, 0]
            if self.labels[id_1] < self.labels[id_2]-self.e:
                label = [1, 0, 0]
            elif self.labels[id_1] > self.labels[id_2]+self.e:
                label = [0, 0, 1]
            # label = -1 if self.labels[id_1] < self.labels[id_2] else 1

        elif self.classes == 2:
            label = [1, 0] if self.labels[id_1] < self.labels[id_2] else [0, 1]
        elif self.classes == 1:
            return self.graphs[idx], self.labels[idx]

        return (self.graphs[id_1], self.graphs[id_2], label)

    def save(self, save_path):
        # save graphs and labels
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        save_graphs(graph_path, list(self.graphs), {
                    'labels': torch.Tensor(self.labels)})
        # save other information in python dict
        info_path = os.path.join(save_path + '_info.pkl')
        save_info(info_path, {'num_classes': self.classes})

    def load(self, save_path):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(save_path + '_info.pkl')
        self.classes = load_info(info_path)['num_classes']

    def has_cache(self, save_path):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        info_path = os.path.join(save_path + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
