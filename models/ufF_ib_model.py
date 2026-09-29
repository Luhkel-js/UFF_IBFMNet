import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap

@MODEL_REGISTRY.register()
class UFFIBModel(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(UFFIBModel, self).__init__(opt)

    def feed_data(self, data):
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
        feat_x_raw = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  
        feat_y_raw = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  

        half_c = feat_x_raw.shape[-1] // 2
        feat_x_inv, feat_y_inv = feat_x_raw[..., :half_c], feat_y_raw[..., :half_c]

        evecs_x, evecs_y = data_x['evecs'], data_y['evecs']
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'], data_y['evecs_trans']
        evals_x, evals_y = data_x['evals'], data_y['evals']

        bidirectional_flag = self.opt['train']['losses']['uff_ib_loss'].get('bidirectional', False)
        if bidirectional_flag:
            Pyx, Pxy = self.compute_permutation_matrix(feat_x_inv, feat_y_inv, bidirectional=True)
        else:
            Pyx = self.compute_permutation_matrix(feat_x_inv, feat_y_inv, bidirectional=False)

        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        diff = evecs_y - torch.bmm(Pyx, torch.bmm(evecs_x, Cxy_est.transpose(-2, -1)))
        loss_geo = torch.linalg.norm(diff)

        w_comm = self.opt['train'].get('w_comm', 0.1)
        w_dir = self.opt['train'].get('w_dir', 0.01)
        loss_comm = loss_dirichlet = torch.tensor(0.0).to(self.device)

        if w_comm > 0:
            alpha = self.opt['train'].get('resolvent_alpha', 5.0)
            R_x_mat = torch.diag_embed(1.0 / (evals_x + alpha)) 
            R_y_mat = torch.diag_embed(1.0 / (evals_y + alpha))
            comm_residual = torch.bmm(Cxy_est, R_x_mat) - torch.bmm(R_y_mat, Cxy_est)
            loss_comm = torch.linalg.norm(comm_residual)

        if w_dir > 0:
            Z_tilde_x = torch.bmm(evecs_trans_x, feat_x_inv)
            Z_tilde_y = torch.bmm(evecs_trans_y, feat_y_inv)
            evals_x_sqrt = torch.sqrt(torch.clamp(evals_x, min=1e-8)).unsqueeze(-1)
            evals_y_sqrt = torch.sqrt(torch.clamp(evals_y, min=1e-8)).unsqueeze(-1)
            loss_dirichlet = (torch.linalg.norm(Z_tilde_x * evals_x_sqrt) + torch.linalg.norm(Z_tilde_y * evals_y_sqrt)) / 2.0

        rfmnet_loss = loss_geo + w_comm * loss_comm + w_dir * loss_dirichlet
        self.loss_metrics = self.losses['rfmnet_loss'](rfmnet_loss)  
        self.loss_metrics['l_comm'] = w_comm * loss_comm
        self.loss_metrics['l_dir'] = w_dir * loss_dirichlet

        if 'ufF_ib_loss' in self.losses:
            self.loss_metrics.update(self.losses['ufF_ib_loss'](feat_x_raw, feat_y_raw))

    def validate_single(self, data, timer):
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
        timer.start()
        
        feat_x_raw = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y_raw = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        half_c = feat_x_raw.shape[-1] // 2
        feat_x = F.normalize(feat_x_raw[..., :half_c], dim=-1, p=2)
        feat_y = F.normalize(feat_y_raw[..., :half_c], dim=-1, p=2)

        evecs_x, evecs_y = data_x['evecs'].squeeze(), data_y['evecs'].squeeze()
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'].squeeze(), data_y['evecs_trans'].squeeze()
        p2p = nn_query(feat_x, feat_y).squeeze()

        if self.non_isometric:
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            for _ in range(5):
                Cxy = evecs_trans_y @ evecs_x[p2p]
                p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        timer.record()
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_y, feat_x.transpose(1, 2))
        Pyx = self.networks['permutation'](similarity)
        if bidirectional:
            Pxy = self.networks['permutation'](similarity.transpose(1, 2))
            return Pyx, Pxy
        else:
            return Pyx

    def refine(self, data):
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True
        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()
        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        avg_error = super(UFFIBModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
        return avg_error