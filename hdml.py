import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, in_channel=128, out_channel=1024):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_channel, in_channel*4)
        self.bn1 = nn.BatchNorm1d(in_channel*4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel*4, out_channel)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        return out


class Generator1(nn.Module):
    def __init__(self, in_channel=128, out_channel=128):
        super(Generator1, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
                # layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channel, 512),
            nn.Dropout(0.1),
            *block(512, 256),
            *block(256, 512),
            nn.Linear(512, in_channel),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class Discriminator1(nn.Module):
    def __init__(self, in_channel=128, out_channel=1):
        super(Discriminator1, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append([nn.LeakyReLU(0.2, inplace=True)])
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            return layers

        self.model = nn.Sequential(
            *block(in_channel, 512),
            *block(512, 256),
            nn.Dropout(0.2),
            *block(256, 256),
            nn.Dropout(0.2),
            nn.Linear(256, out_channel),
            nn.Sigmoid(),
        )

    def forward(self, embedding):
        # Concatenate label embedding and image to produce input
        validity = self.layer(embedding)
        return validity

class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, in_channel=128, out_channel=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.LeakyReLU(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, out_channel),
            nn.Sigmoid() if out_channel==1 else Identity(),
        )

    def forward(self, embedding):
        # Concatenate label embedding and image to produce input
        validity = self.layer(embedding)
        return validity


class TripletPulling(nn.Module):
    def __init__(self, embedding_size=128, alpha=5.0):
        super(TripletPulling, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha

    def forward(self, x, jm):
        # if isinstance(jm, torch.Tensor):
        #     jm = jm.item()
        # jm = 1e-7
        lmd = np.exp(-self.alpha / jm)
        # lmd = 0.5
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        # r = ((dist_pos + (dist_neg - dist_pos) * np.exp(-self.alpha / jm)) / dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        r = (lmd + (1- lmd) * dist_pos/dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        # r = ((dist_pos * (1 - np.exp(-self.alpha / jm)) + dist_neg * np.exp(-self.alpha / jm)) / dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        # print((dist_pos + (dist_neg - dist_pos) * np.exp(-self.alpha / jm))/ dist_neg)
        # import ipdb; ipdb.set_trace()
        neg2 = anchor + torch.mul((negative - anchor), r)
        neg_mask = torch.ge(dist_pos, dist_neg)
        op_neg_mask = ~neg_mask
        neg_mask = neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        op_neg_mask = op_neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        neg_hat = torch.mul(negative, neg_mask) + torch.mul(neg2, op_neg_mask)

        # r2 = ((1- lmd) * (dist_neg - dist_pos)/dist_pos).unsqueeze(-1).repeat(1, self.embedding_size)
        # pos2 = positive + torch.mul((positive - anchor), r2)
        # pos_mask = torch.ge(dist_pos, dist_neg)
        # op_pos_mask = ~pos_mask
        # pos_mask = pos_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        # op_pos_mask = op_pos_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        # pos_hat = torch.mul(positive, pos_mask) + torch.mul(pos2, op_pos_mask)
        # # (F.pairwise_distance(anchor, pos_hat, 2)-dist_pos).sum()/(op_pos_mask.sum()/self.embedding_size)

        return torch.cat([anchor, positive, neg_hat], dim=0), (F.pairwise_distance(anchor, neg_hat, 2)-dist_neg).sum()/(op_neg_mask.sum()/self.embedding_size)


class TripletPullingPos(nn.Module):
    def __init__(self, embedding_size=128):
        super(TripletPullingPos, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, x, dis_mean):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        neg_mask = torch.ge(dist_pos, dis_mean)
        op_neg_mask = ~neg_mask
        neg_mask = neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        op_neg_mask = op_neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)

        lam = torch.exp(dis_mean-dist_pos).view([-1, 1])*neg_mask + (1+(dis_mean-dist_pos)*0.3).view([-1, 1])*op_neg_mask #TODO        
        # lam = 1.0*neg_mask + (1.3*(dis_mean-dist_pos)/dist_pos).view([-1, 1])*op_neg_mask #TODO        
        anchor = anchor + torch.mul(anchor - positive, lam)
        positive = positive + torch.mul(positive - anchor, lam)
        return torch.cat([anchor, positive, negative], dim=0), dist_pos.mean()


class Positivate_Adapt_Pulling(nn.Module):
    def __init__(self, embedding_size=128, alpha=90.0):
        super(TripletPulling, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha

    def forward(self, x, jm):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        # r = (dist_pos + (dist_neg - dist_pos) * np.exp(-self.alpha / jm) / dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        r = ((dist_pos + (dist_neg - dist_pos) * np.exp(-self.alpha / jm)) / dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        neg2 = anchor + torch.mul((negative - anchor), r)
        neg_mask = torch.ge(dist_pos, dist_neg)
        op_neg_mask = ~neg_mask
        neg_mask = neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        op_neg_mask = op_neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        neg_hat = torch.mul(negative, neg_mask) + torch.mul(neg2, op_neg_mask)
        return torch.cat([anchor, positive, neg_hat], dim=0)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()


class RevTripletLoss(nn.Module):
    def __init__(self):
        super(RevTripletLoss, self).__init__()

    def forward(self, x, margin):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(-dist_pos + dist_neg + margin)
        return losses.mean()
