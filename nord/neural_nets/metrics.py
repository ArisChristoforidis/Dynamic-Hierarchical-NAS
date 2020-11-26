import numpy as np
import torch
from nord.utils import rank_inds_of_lists
from scipy.stats import kendalltau, rankdata, spearmanr


def accuracy(predicted, targets):
    return (predicted == targets).float().mean()


def one_hot_accuracy(predicted, targets):
    predicted = torch.stack(predicted)
    targets = torch.stack(targets)
    acc = (predicted.max(dim=1)[1] == targets.max(
        dim=1)[1]).float().mean().cpu().item()
    return {'acc': acc}


def binary_rank_correlation_tau_with_top(p, metric='kendall'):
    chosen_metric = kendalltau
    if metric == 'spearman':
        chosen_metric = spearmanr

    def ktwt(predicted, targets, all_to_all=True):
        predicted = torch.stack(predicted)
        targets = torch.stack(targets)
        sz = len(predicted)
        all_inds = [i for i in range(sz)]

        original_ranks = rank_inds_of_lists(targets, all_inds, all_to_all)
        calculated_ranks = rank_inds_of_lists(
            predicted, all_inds, all_to_all)

        all_t = chosen_metric(original_ranks, calculated_ranks)

        top_half_inds = np.argsort(original_ranks)[-int(sz*p):]
        original_top_ranks = rank_inds_of_lists(
            targets, top_half_inds, all_to_all)
        calculated_top_ranks = rank_inds_of_lists(
            predicted, top_half_inds)
        top_t = chosen_metric(original_top_ranks,
                              calculated_top_ranks, all_to_all)

        return {metric: all_t, 'top_'+metric: top_t}

    return ktwt


def regression_rank_correlation_with_top(p, metric='kendall'):
    chosen_metric = kendalltau
    if metric == 'spearman':
        chosen_metric = spearmanr

    def ktwt(predicted, targets, all_to_all=True):
        predicted = torch.stack(predicted)
        targets = torch.stack(targets)
        original_ranks = rankdata((targets.cpu().reshape((-1, ))*333).round())
        calculated_ranks = rankdata(
            (predicted.cpu().reshape((-1, ))*333).round())

        all_t = chosen_metric(original_ranks, calculated_ranks)

        sz = len(original_ranks)
        top_half_inds = np.argsort(original_ranks)[-int(sz*p):]
        original_top_ranks = rankdata(
            targets.cpu().reshape((-1, ))*333).round()[top_half_inds]
        calculated_top_ranks = rankdata(
            predicted.cpu().reshape((-1, ))*333).round()[top_half_inds]
        top_t = chosen_metric(original_top_ranks, calculated_top_ranks)

        return {metric: all_t, 'top_'+metric: top_t}

    return ktwt
