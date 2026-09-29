import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)



@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=0.0, w_cont=0.0, bidirectional=True):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0 and w_cont >=0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.w_cont = w_cont
        self.bidirectional = bidirectional
    def forward(self, P12, P21, C12, C21, C11, C22, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = SquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)
        zeros = torch.zeros_like(C12)
        losses = dict()

        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) 
            if self.bidirectional:
                bijectivity_loss += criterion(torch.bmm(C21, C12), eye_batch) 
            bijectivity_loss *= self.w_bij
            losses['l_bij'] = bijectivity_loss

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)  
            if self.bidirectional:
                orthogonality_loss += criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) 
            orthogonality_loss *= self.w_orth
            losses['l_orth'] = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            resolvant_gamma = 0.5
            D12 = get_mask(evals_1, evals_2, resolvant_gamma)  
            D21 = get_mask(evals_2, evals_1, resolvant_gamma) 
            laplacian_loss = criterion(C12 * D12 , zeros)
            if self.bidirectional:
                laplacian_loss += criterion(C21 * D21 , zeros) 
            laplacian_loss *= self.w_lap
            losses['l_lap'] = laplacian_loss

        if self.w_cont > 0:
            contrast_loss = criterion(C22, eye_batch)   
            if self.bidirectional:
                contrast_loss += criterion(C11, eye_batch)
            contrast_loss *= self.w_cont
            losses['l_self'] = contrast_loss

        return losses



@LOSS_REGISTRY.register()
class RfmnetLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(RfmnetLoss, self).__init__()
        assert loss_weight >= 0
        self.loss_weight = loss_weight

    def forward(self, loss):
        losses = dict()
        losses['rfmnet'] = self.loss_weight * loss
        return losses
            

# @LOSS_REGISTRY.register()
# class ContrastiveLoss(nn.Module):
#     def __init__(self, w_self=1.0, w_other=1.0, bidirectional=True):
#         super(ContrastiveLoss, self).__init__()
#         assert w_self >= 0 and w_other >= 0 
#         self.w_self = w_self
#         self.w_other = w_other
#         self.bidirectional = bidirectional

#     def forward(self, C12, C21, C11, C22, C12_p, C21_p, C11_p, C22_p):

#         criterion = SquaredFrobeniusLoss()
#         losses = dict()

#         if self.w_other > 0:
#             other_contrastive_loss = criterion(C12_p, C12)
#             if self.bidirectional:
#                 other_contrastive_loss += criterion(C21_p, C21)
#             other_contrastive_loss *= self.w_other
#             losses['l_other_align'] = other_contrastive_loss

#         if self.w_self > 0:
#             self_contrastive_loss = criterion(C11_p, C11)
#             if self.bidirectional:
#                 self_contrastive_loss += criterion(C22_p, C22)
#             self_contrastive_loss *= self.w_self
#             losses['l_self_align'] = self_contrastive_loss

#         return losses


@LOSS_REGISTRY.register()
class PartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialFmapsLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = SquaredFrobeniusLoss()
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        # compute area ratio between full shape and partial shape r
        r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1)
        eye = torch.zeros_like(C_fp)
        eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * criterion(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_bij * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss}
