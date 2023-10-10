import torch
from offpolicy.algorithms.qmix.algorithm.q_mixer import QMixer

class ROMAMixer(QMixer):
    """See parent class."""

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        super(ROMAMixer, self).__init__(args, num_agents, cent_obs_dim, device, multidiscrete_list=multidiscrete_list)