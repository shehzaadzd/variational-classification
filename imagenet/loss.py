import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class Discriminators_2l(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(Discriminators_2l, self).__init__() 
        self.hidden_dim = 512
        self.W1 = nn.Parameter(torch.randn(num_classes, self.hidden_dim, feat_dim))
        self.b1 = nn.Parameter(torch.zeros(num_classes, self.hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_classes, self.hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        self.relu  = nn.Tanh()

    def forward(self, Z, y):
        w1 = self.W1[y, :, :] #B x 2*feat_dim x feat_dim
        w2 = self.W2[y, :] #B x 2* feat_dim x 1
        b1 = self.b1[y, :] # B x 2* feat
        b2 = self.b2[y, ] # B x 1
        op = self.relu((w1 * Z.unsqueeze(1)).sum(-1) + b1)
        op =(w2*op).sum(-1) + b2
        return op
    def reset(self,):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.zeros_(self.b1)
        torch.nn.init.xavier_uniform_(self.W2)
        torch.nn.init.zeros_(self.b2)






class Discriminators_1l(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(Discriminators_1l, self).__init__() 
        self.W1 = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.b1 = nn.Parameter(torch.zeros(num_classes))
        # self.dropout = nn.Dropout(0.2)
    def forward(self, Z, y):
        w1 = self.W1[y, :] #B x feat_dim x feat_dim
        b1 = self.b1[y] # B x feat
        op = (w1 * Z).sum(-1) + b1
        return op


class VariationalClassification(nn.Module):
    def __init__(self, args, num_classes, feat_dim , ) -> None:
        super(VariationalClassification, self).__init__()
        self.num_classes = num_classes
        self.NLL = nn.CrossEntropyLoss()
        self.VC = AdversarialContrastiveLoss(num_classes, feat_dim, alpha=0.0)

        self.classifier = Discriminators_2l(num_classes=num_classes, feat_dim=feat_dim) if args.disc_layers == 2 else Discriminators_1l(num_classes=num_classes, feat_dim=feat_dim)

        self.disc_optimizer = torch.optim.AdamW( [p for p in self.classifier.parameters()] )

        self.args = args
    def forward(self,outputs, targets ):
        args = self.args
        logits, self.real, self.sampled, likelihood = self.VC(outputs, label= targets)
        self.targets = targets
        Treal = self.classifier(outputs, targets).squeeze(-1)
        loss_1 = self.NLL(logits, targets)  + args.l1*(likelihood.mean()) 
        loss_2 = Treal.mean()
        loss = loss_1 + args.l2*loss_2
        return loss, logits

    def discriminator_train(self):
        Tsampled = self.classifier(self.sampled, self.targets).squeeze(-1)
        Treal = self.classifier(self.real, self.targets).squeeze(-1)
        self.disc_optimizer.zero_grad()
        dual_loss = ( F.binary_cross_entropy_with_logits(Treal, torch.ones_like(Treal)) + F.binary_cross_entropy_with_logits(Tsampled, torch.zeros_like(Tsampled)) )
        dual_loss.backward()
        self.disc_optimizer.step()
        self.disc_optimizer.zero_grad()


class LGM(nn.Module):
    def __init__(self, args, num_classes, feat_dim , ) -> None:
        super(LGM, self).__init__()
        self.NLL = nn.CrossEntropyLoss()
        self.VC = AdversarialContrastiveLoss(num_classes, feat_dim, alpha=0.0)

        self.args = args
    def forward(self,outputs, targets ):
        args = self.args
        logits, self.real, self.sampled, likelihood = self.VC(outputs, label= targets, )
        self.targets = targets

        loss_1 = self.NLL(logits, targets)  + args.l1*(likelihood.mean()) 
        loss = loss_1 
        return loss, logits



class AdversarialContrastiveLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """

    def __init__(self, num_classes, feat_dim , alpha):
        super(AdversarialContrastiveLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        #Theta
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label=None):

        batch_size = feat.shape[0]
        feat_dim = feat.shape[1]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)

        covs = torch.exp(log_covs)  # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)  # n*c*d

        diff = torch.unsqueeze(feat, dim=1) - \
            torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)  # eq.(18)


        slog_covs = torch.sum(log_covs, dim=-1)  # 1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        logits = -0.5 * (tslog_covs + dist)
        likelihood_logits = logits

        if label == None:
            label = torch.argmin(dist, dim=-1)

        Treal = None
        Tsampled = None
        if label != None:
            Treal = feat 
            Tsampled = (torch.Tensor(batch_size, feat_dim).normal_().cuda() * torch.sqrt(torch.exp(self.log_covs[label])) + self.centers[label].cuda())
        likelihood = -likelihood_logits[torch.arange(batch_size), label]
        return logits, Treal.detach(), Tsampled.detach(), likelihood
