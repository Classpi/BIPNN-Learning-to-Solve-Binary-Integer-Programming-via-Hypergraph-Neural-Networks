import torch
from tqdm import tqdm

def loss_pubo(outs, H, C, **kwargs):
    H = kwargs.get("pubo_H", H)
    epoch = kwargs.pop("epoch", 1)
    annl_cof_l = kwargs.pop("acl", lambda e: 0)
    annl_cof = annl_cof_l(epoch)

    H0 = outs * H
    H1 = torch.where(H == 0, torch.tensor(1.0), H0)
    H2 = torch.prod(H1, dim=0).unsqueeze(0)
    loss = torch.mul(H2, C).sum()

    annl = _loss_gini(outs)
    if epoch % 200 == 0:
        tqdm.write(f"epoch: {epoch}: loss:{loss:.2f} | annl: {annl:.2f}")
    return loss + annl_cof * annl


def _loss_gini(outs):
    annl = (1 - (2 * outs - 1).pow(2)).sum()
    return annl


def loss_pubo_iteration(outs, H, C, **kwargs):
    loss = 0.0
    num_edges = H.shape[1]
    for i in range(num_edges):
        h_col = H[:, i]
        h0_i = outs * h_col
        h1_i = torch.where(h_col == 0, torch.tensor(1.0, device=outs.device), h0_i)
        edge_product = torch.prod(h1_i)
        loss += C[i] * edge_product

    epoch = kwargs.pop("epoch", 1)
    annl_cof_l = kwargs.pop("acl", lambda e: 0)
    annl_cof = annl_cof_l(epoch)
     
    annl = _loss_gini(outs)
     
    if epoch % 100 == 0:
        tqdm.write(f"epoch: {epoch}: loss:{loss:.2f} | annl: {annl:.2f}")
    
    return loss + annl_cof * annl
