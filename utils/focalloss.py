from jittor import nn
import jittor as jt

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce =jt.nn.CrossEntropyLoss()

    def execute(self, input, target):
        logp = self.ce(input, target)
        p = jt.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()