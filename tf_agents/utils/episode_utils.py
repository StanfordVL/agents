from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle


def get_metrics(episodes):
    success_rate = np.mean([episode.success for episode in episodes])
    # spl = np.mean([episode.success * episode.path_efficiency for episode in episodes])
    # kinematic_disturbance = np.mean([episode.success * episode.kinematic_disturbance for episode in episodes])
    # dynamic_disturbance_a = np.mean([episode.success * episode.dynamic_disturbance_a for episode in episodes])
    # dynamic_disturbance_b = np.mean([episode.success * episode.dynamic_disturbance_b for episode in episodes])
    # collision_step = np.mean([episode.collision_step for episode in episodes])
    return {
        'success_rate': success_rate,
        # 'spl': spl,
        # 'kinematic_disturbance': kinematic_disturbance,
        # 'dynamic_disturbance_a': dynamic_disturbance_a,
        # 'dynamic_disturbance_b': dynamic_disturbance_b,
        # 'collision_step': collision_step
    }


def save(episodes, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(episodes, f)
