import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class UFFIBLoss(nn.Module):
    def __init__(self, w_inter=1.0, w_intra=0.1, w_ib=0.5, tau_inter=0.1, tau_intra=0.1, inter_top_k=30, intra_top_k=30, bidirectional=True, intra_repulsion_only=False):
        super(UFFIBLoss, self).__init__()
        self.w_inter = w_inter
        self.w_intra = w_intra
        self.w_ib = w_ib          
        self.tau_inter = tau_inter
        self.tau_intra = tau_intra
        self.inter_top_k = inter_top_k
        self.intra_top_k = intra_top_k
        self.bidirectional = bidirectional
        self.intra_repulsion_only = intra_repulsion_only 

    def margin_infonce(self, z_inv_x, z_inv_y, tau, top_k, repulsion_only=False):
        """
        InfoNCE (Margin-InfoNCE)
        Boolean Mask
        """
        B, N, C = z_inv_x.shape
        sim = torch.bmm(z_inv_x, z_inv_y.transpose(1, 2)) / tau
        
        vals, idx = sim.topk(k=top_k, dim=-1)
        
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask.scatter_(-1, idx, False)
        
        sim_neg = sim[mask].view(B, N, -1) 
        
        neg_term = torch.logsumexp(sim_neg, dim=-1) # [B, N]

        if repulsion_only:
            return torch.mean(neg_term)
        else:
            #InfoNCE
            pos_term = vals.mean(dim=-1) # [B, N]
            return torch.mean(-pos_term + neg_term)

    def forward(self, feat_x, feat_y, curr_epoch=0):
        losses = dict()
        
        # cut(64 / 64)
        half_c = feat_x.shape[-1] // 2
        z_inv_x, z_var_x = feat_x[..., :half_c], feat_x[..., half_c:]
        z_inv_y, z_var_y = feat_y[..., :half_c], feat_y[..., half_c:]

        z_inv_x = F.normalize(z_inv_x, p=2, dim=-1)
        z_inv_y = F.normalize(z_inv_y, p=2, dim=-1)
        z_var_x_norm = F.normalize(z_var_x, p=2, dim=-1)
        z_var_y_norm = F.normalize(z_var_y, p=2, dim=-1)

        # 1. IB-loss
        if self.w_ib > 0:
            corr_x = torch.bmm(z_inv_x.transpose(1, 2), z_var_x_norm) 
            corr_y = torch.bmm(z_inv_y.transpose(1, 2), z_var_y_norm)
            losses['l_ib'] = self.w_ib * (torch.mean(corr_x ** 2) + torch.mean(corr_y ** 2))

            std_x = torch.sqrt(z_var_x.var(dim=1) + 1e-4).mean()
            std_y = torch.sqrt(z_var_y.var(dim=1) + 1e-4).mean()
            losses['l_var_std'] = self.w_ib * 0.5 * (torch.relu(1.0 - std_x) + torch.relu(1.0 - std_y))

        # 2.Inter-shape-loss
        if self.w_inter > 0:
            inter_loss = self.margin_infonce(z_inv_x, z_inv_y, self.tau_inter, self.inter_top_k, repulsion_only=False)
            if self.bidirectional:
                inter_loss += self.margin_infonce(z_inv_y, z_inv_x, self.tau_inter, self.inter_top_k, repulsion_only=False)
            losses['l_inter'] = self.w_inter * inter_loss

        # 3. Intra-shape-loss
        if self.w_intra > 0:
            intra_loss = self.margin_infonce(z_inv_y, z_inv_y, self.tau_intra, self.intra_top_k, repulsion_only=self.intra_repulsion_only)
            if self.bidirectional:
                intra_loss += self.margin_infonce(z_inv_x, z_inv_x, self.tau_intra, self.intra_top_k, repulsion_only=self.intra_repulsion_only)
            losses['l_intra'] = self.w_intra * intra_loss

        return losses