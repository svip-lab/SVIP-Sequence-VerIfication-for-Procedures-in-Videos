import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss



def compute_cls_loss(pred, labels, use_cosface=False):

    if use_cosface:
        # CosFace Loss

        s = 30.0
        m = 0.4

        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        # Softmax Loss

        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    return loss


def compute_seq_loss(seq1, seq2):


    if seq1 == None or seq2 == None:
        return 0


    seq1 = F.normalize(seq1, 2, dim=2)
    seq2 = F.normalize(seq2, 2, dim=2)

    bs, length, _ = seq1.size()

    corr = torch.bmm(seq1, seq2.transpose(1, 2))
    corr1 = nn.Softmax(dim=1)(corr)  # Softmax across columns
    corr2 = nn.Softmax(dim=2)(corr)  # Softmax across rows
    corr = (corr1 + corr2) / 2

    sims = torch.diagonal(corr, dim1=1, dim2=2)

    loss = torch.sum(torch.tensor(1) - sims) / bs

    return loss

