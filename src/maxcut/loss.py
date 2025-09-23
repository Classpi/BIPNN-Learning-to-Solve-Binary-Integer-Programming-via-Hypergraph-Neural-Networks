import torch
from tqdm import tqdm


def loss_max_cut(outs, H, **kwargs):
    r"""Use the max-cut objective function on a hypergraph to simulate a PUBO problem

    The max-cut problem can be expressed as: O = \sum_{e\in E} \big( 1- \prod_{i \in e}x_i - \prod_{i \in e}(1-x_i)  \big),
    It describes a polynomial with a one-time term.

    Args:
        outs: shape (V, 1),
        H: shape (V, E),

    Notes:
    -----
    - This loss function should be used in one-dimensional {0, 1} problems,
      which means arg `outs`'s shape should be `(V, 1)`
    - And please choose `src.loss.loss_max_cut_onehot` for multi-dimensional tasks
    """
    epoch = kwargs.pop("epoch", 1)
    annl_cof_l = kwargs.pop("acl", lambda e:0)
    annl_cof = annl_cof_l(epoch)

    H = H.to_dense()
    mask_outs = (outs * H + (torch.ones_like(H) - H)).prod(dim=0)  # Set the irrelevant position to 1 so that it cannot participate in multiplication

    temp2 = ((torch.ones_like(outs) - outs) * H + torch.ones_like(H) - H).prod(dim=0)

    O = (torch.ones((1, outs.shape[1])) - mask_outs - temp2).sum()
    loss = -1 * O
    annl = _loss_gini(outs)
    if epoch % 100 == 0:
        tqdm.write(f"epoch: {epoch}: loss:{loss:.2f} | annl: {annl:.2f}")
    return loss + annl_cof * annl

def _loss_gini(outs):
    annl = (1 - (2 * outs - 1).pow(2)).sum()
    return annl


def _loss_max_cut_onehot(outs, H):
    r"""
    ---
    **WARNING!**
    **This function has never been used or tested in the original paper**
    ---
    Extended the maxcut problem to multi-dimensional 2-dim one-hot
    Args:
        outs: shape (V, 2),
        H: shape (V, E),

    """
    X_ = outs.t().unsqueeze(-1)
    H_ = H.unsqueeze(0)
    weight = H.sum(dim=0)
    mid = X_.mul(H_)
    sum = (mid * (1 / weight)).sum()
    sub = (mid + (1 - H)).prod(dim=1).sum()  # Set the irrelevant position to 1 so that it cannot participate in multiplication
    loss = sum - sub
    return -loss
