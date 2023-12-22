import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class Model(nn.Module):
    """Head for sentence-level classification tasks."""
    '''
    dropout：这是一个 Dropout 层，用于在训练过程中随机关闭一部分神经元，以防止过拟合1。
    out_proj：这是一个线性层，用于将特征转换为输出。
    '''
    def __init__(self, config):
        super().__init__()
        # dropout 随机关闭一些层，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.feature_size, 1)

    def forward(self, features, labels=None):
        x = features.float()
        x = self.dropout(x)
        logits = self.out_proj(x)

        prob = torch.sigmoid(logits)
        if labels is not None:

            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob
        else:
            return prob
        return x
