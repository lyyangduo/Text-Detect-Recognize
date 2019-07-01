import torch 
import math
from torch import nn
from scipy.spatial import distance
from sklearn.preprocessing import normalize 
from scipy.special import binom
import time 

class masked_crossentropy(nn.Module):
    """docstring for masked_crossentropy"""
    def __init__(self,device):
        super(masked_crossentropy, self).__init__()
        self.device = device
        
    def forward(self,logits,target,masks,max_len):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = nn.functional.log_softmax(logits_flat,dim=1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1).to(self.device)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        losses = losses * masks.float()

        length = masks.nonzero().shape[0]
        loss = losses.sum() / length


        return loss
        

class masked_regression(nn.Module):
    def __init__(self,device):
        super(masked_regression, self).__init__()
        self.device = device
        
    def forward(self,preds,labels,masks,max_len):

        """
        Args:
            pred: weighted feature predicted by the model
                (batch, max_len, hidden_size) 
            target: A Variable containing a LongTensor of size
                (batch, number_base_fonts, max_len, hidden_size) 
            masks: binary 
                (batch,max_len)
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_stack: (batch, number_base_fonts, max_len, hidden_size)
        preds_stack = torch.stack([preds]*labels.shape[1],dim=1)
        # labels (batch, number_base_fonts, max_len, hidden_size)
        labels = labels.to(self.device)
        # l2_dist: [batch,number_base_fonts,max_len]
        l2_dist = torch.norm (preds_stack - labels,dim=3, p =2)
        # losses: [batch,max_len]
        losses,_ = torch.min(l2_dist,dim =1)
        #losses = torch.mean(l2_dist,dim=1)
        losses = losses * masks.float()

        length = masks.nonzero().shape[0]
        loss = losses.sum() / length

        return loss




class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, device, centers,num_classes=37, feat_dim=512):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        if centers is not None :
            self.centers = centers 
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))


    def forward(self, outputs, padded_labels,masks):
        """
        Args:
            outputs: [batch_size,max_len,feat_dim]
            padded_labels:[batch_size,max_len]
            masks: [batch_size,max_len]

            x: feature matrix with shape (batch_size, feat_dim). 
            labels: ground truth labels with shape (batch_size). 
        """
        batch_size = outputs.size(0)*outputs.size(1)
        x = outputs.view(-1,self.feat_dim)
        labels = padded_labels.view(-1)
        masks = masks.view(batch_size,1).expand(batch_size,self.num_classes)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())  # (x-y)^2 

        distmat = distmat * masks

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat*mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum()/x.shape[0]

        # dist = []
        # for i in range(batch_size):
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()
        return loss

        




