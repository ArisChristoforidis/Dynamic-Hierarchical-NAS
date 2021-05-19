import torch.nn as nn
from nord.neural_nets.losses import My_MRLoss, My_CrossEntropyLoss
from nord.neural_nets.metrics import accuracy, one_hot_accuracy
from nord.neural_nets.metrics import binary_rank_correlation_tau_with_top as br
from nord.neural_nets.metrics import regression_rank_correlation_with_top as rr


NUM_CLASSES = {'cifar10': 10,
               'activity_recognition': 7,
               'earthquakes': 2,
               'fashion-mnist': 10,
               'graph-nasbench3': 3,
               'graph-nasbench1': 1}

INPUT_SHAPE = {'cifar10': (32, 32),
               'activity_recognition': (52,),
               'earthquakes': (512, ),
               'fashion-mnist': (28, 28),
               'graph-nasbench3': (5, ),
               'graph-nasbench1': (5, )}

CHANNELS = {'cifar10': 3,
            'activity_recognition': 3,
            'earthquakes': 1,
            'fashion-mnist': 1,
            'graph-nasbench2': 5,
            'graph-nasbench3': 5,
            'graph-nasbench1': 5}

CRITERION = {'cifar10': nn.CrossEntropyLoss,
             'activity_recognition': My_CrossEntropyLoss,
             'earthquakes': My_CrossEntropyLoss,
             'fashion-mnist': nn.CrossEntropyLoss,
             'graph-nasbench3': nn.KLDivLoss,
             'graph-nasbench2': nn.KLDivLoss,
             'graph-nasbench1': My_MRLoss}

PROBLEM_TYPE = {'cifar10': 'classification',
                'activity_recognition': 'classification',
                'earthquakes': 'classification',
                'fashion-mnist': 'classification',
                'graph-nasbench2': 'ranking',
                'graph-nasbench3': 'ranking',
                'graph-nasbench1': 'ranking'}

METRIC = {'cifar10':  [accuracy],
          'activity_recognition': [one_hot_accuracy],
          'earthquakes':  [accuracy],
          'fashion-mnist':  [accuracy],
          'graph-nasbench2': [br(0.5), br(0.5, 'spearman')],
          'graph-nasbench3': [br(0.5), br(0.5, 'spearman')],
          'graph-nasbench1': [rr(0.5), rr(0.5, 'spearman')]}

DIMENSION_KEEPING = {'cifar10': 32,
                     'fashion-mnist': None,
                     'graph-nasbench3': None,
                     'graph-nasbench1': None,
                     'activity_recognition': None,
                     'earthquakes': None}

