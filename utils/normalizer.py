# utils/reward_normalizer.py
import torch


class RewardNormalizer:
    def __init__(self, dim=4, momentum=0.01, eps=1e-6, l2_normalize=True, device="cpu"):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 0
        self.momentum = momentum
        self.eps = eps
        self.l2_normalize = l2_normalize

    def update(self, reward_vec):
        self.count += 1
        self.mean = (1 - self.momentum) * self.mean + self.momentum * reward_vec
        self.var = (1 - self.momentum) * self.var + self.momentum * (
            reward_vec - self.mean
        ) ** 2

    def normalize(self, reward_vec):
        normed = (reward_vec - self.mean) / (torch.sqrt(self.var) + self.eps)
        if self.l2_normalize:
            normed = normed / (normed.norm() + self.eps)
        return normed
