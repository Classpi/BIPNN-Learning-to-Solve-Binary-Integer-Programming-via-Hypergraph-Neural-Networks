from typing import Tuple
import torch
from tqdm import tqdm
from src.constraint import BaseConstraintGenerator
from src.utils import get_obj_value,_gumbel_sample
import torch.nn.functional as F

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

def loss_pubo_constraint(outs,
                         H,
                         C,
                         constraints,
                         mu,
                         num_samples: int = 10,
                         epoch: int = 1,
                         ) -> Tuple[torch.Tensor, float, float, float, bool]:
    
    mu_step_size = 1
    mu_value = 0.00001
    mu_min = 0.
    mu_max = float('inf')
    annl_conf = 10.
    num_samples = 10
    x = _gumbel_sample(outs, num_samples).float().reshape(num_samples, -1)
    cons_pos = torch.relu(constraints.calculate_slack(x.T)).mean(dim=1, keepdim=True)
    mu = mu + mu_step_size * (cons_pos.sum().item()/constraints.m - mu_value)
    mu = max(min(mu, mu_max), mu_min)
    x_=torch.sigmoid(outs)
    H0 = x_* H
    H1 = torch.where(H == 0, torch.tensor(1.0, device=H.device), H0)
    H2 = torch.prod(H1, dim=0).unsqueeze(0)
    obj = torch.mul(H2, C).sum()
    cons_pos1 =  cons_pos
    loss_cons = obj + mu * cons_pos.mean()*H.shape[1]
    best_obj_val = float('inf')
    best_satisfied_rate = 0.0
    has_legal_sol = False
    with torch.no_grad():
        if epoch % 200 == 0:
            tqdm.write(f"epoch: {epoch}: loss:{loss_cons:.2f} | obj: {obj:.2f} | mu: {mu:.2f} | cons_pos:{cons_pos1.mean()}")
            obj_values_list = []          
            single_satisfied_rate_list = [] 
            legal_flag_list = []          
            total_sample = 500          
            total_constraint = constraints.m
        if epoch % 1000 == 0:
            for _ in range(total_sample):
                xx = _gumbel_sample(outs, 1, 1.0).float().transpose(0, 1)
                obj_val = get_obj_value(H, C, xx).item()

                slack_val = constraints.calculate_slack(xx)
                is_legal = torch.all(torch.le(slack_val, 0)).item()
                single_satisfied = torch.le(slack_val, 0).float()
                single_satisfied_rate = single_satisfied.mean().item() * 100

                obj_values_list.append(obj_val)
                single_satisfied_rate_list.append(single_satisfied_rate)
                legal_flag_list.append(is_legal)

            obj_values_validate = torch.tensor(obj_values_list)
            single_sample_satisfied_rate = torch.tensor(single_satisfied_rate_list)
            legal_flags = torch.tensor(legal_flag_list)
    
            idx_legal = torch.where(legal_flags == True)[0]
            legal_sample = len(idx_legal)
            best_satisfied_rate = max(single_sample_satisfied_rate.max().item(), best_satisfied_rate)
            idx_max_rate = torch.where(single_sample_satisfied_rate == best_satisfied_rate)[0]
            best_obj_val = obj_values_validate[idx_max_rate].min().item()
            has_legal_sol = best_satisfied_rate >= 100.0 
    return loss_cons, mu, best_obj_val, best_satisfied_rate, has_legal_sol
