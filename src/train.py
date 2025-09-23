from typing import Tuple
import torch
import logging

from .loss import loss_pubo
from .models import Net
from .utils import from_hypergraph_to_graph_clique
from .maxcut import maxcut_evaluate

from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(net, X, graph, optimizer, loss_fn, clip_grad=False, **kwargs):
    """Conduct a round of neural network training

    Args:
        kwargs:
            `loss_fn`'s kwargs

    """
    net.train()
    optimizer.zero_grad()
    outs = net.forward(X, graph, **kwargs)
    loss: torch.Tensor = loss_fn(outs, **kwargs)
    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), max_norm=5.5)
    optimizer.step()
    return loss.detach().item(), outs


def run(layers, X, graph, num_epochs, loss_fn, lr, log=False, **kwargs) -> Tuple[float, torch.Tensor]:
    log_info = {
        "epoch": [],
        "cut_edges": [],
        "not_converged": [],
    }
    net = Net(layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-8)
    for epoch in tqdm(range(1, num_epochs + 1)):
        kwargs = {**kwargs, "epoch": epoch}
        loss, outs = train(net, X, graph, optimizer, loss_fn, **kwargs)
        if log and epoch % 20 == 0:
            o_g = kwargs.get("o_graph", None)
            dic = maxcut_evaluate(outs, o_g)
            log_info["epoch"].append(epoch),
            log_info["cut_edges"].append(dic["cut_edges"]),
            log_info["not_converged"].append(dic["not_converged"])
    del net, optimizer
    if log:
        import pandas as pd
        import time

        pd.DataFrame(log_info).to_csv(f"/home/exs/work/binary-programming/tmp/log/max_cut_log_{time.asctime()}", index=False)
    return loss, outs


def run_pubo(layers, hg, H, C, X, num_epochs, lr, convert=False, **kwargs):
    """
    Run PUBO training.

    Args:
        layers: Network architecture.
        H: Hypergraph matrix.
        C: Coefficient matrix.
        X: Input data.
        num_epochs: Number of training epochs.
        lr: Learning rate for optimizer.  # 修改点8: 添加参数说明
    """
    if convert:
        g = from_hypergraph_to_graph_clique(hg)
        edge_index = torch.tensor(g.e[0], dtype=torch.long).t().contiguous()
    else:
        g = hg
        edge_index = None
    return run(layers, X, g, num_epochs, loss_pubo, lr=lr, H=H, C=C, edge_index=edge_index, **kwargs)
