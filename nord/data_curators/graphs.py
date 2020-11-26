import numpy as np
import torch
from torch.utils.data import DataLoader

from .graph_data.nasbench_graphs import get_nasbench_dataset
from nord.utils import graph_collate, graph_pair_collate


def get_nasbench_graphs(batch_size, initial_pop, mutated, classes, val_perc,
                        test_perc, all_vs_all=[False, False]):

    np.random.seed(1337)
    torch.random.manual_seed(1337)

    data = get_nasbench_dataset(initial_pop, mutated, classes,  1337, val_perc,
                                test_perc, all_vs_all)

    train, validation, test = data

    train_loader, test_loader = None, None
    if classes > 1:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_pair_collate)

        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_pair_collate)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_collate)

        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_collate)

    return train_loader, test_loader, classes
