from offpolicy.algorithms.qmix.qmix import QMix
import torch

class ROMA(QMix):
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cuda:0"), episode_length=None):
        """See parent class."""
        super(ROMA, self).__init__(args, num_agents, policies, policy_mapping_fn, device=device, episode_length=episode_length, vdn=True)
