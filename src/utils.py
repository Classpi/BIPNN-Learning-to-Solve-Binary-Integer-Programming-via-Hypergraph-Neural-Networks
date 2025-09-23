from copy import deepcopy
import os
from typing import Any
import pickle as pkl

import dhg
import dimod
import torch
import logging
import numpy as np

from dhg import Graph, Hypergraph
from neal import SimulatedAnnealingSampler
from pyqubo import Array
from pyscipopt import Model, quicksum
from pyscipopt.recipes import nonlinear

logger = logging.getLogger(__name__)


# PUBO utils
def generate_hypergraph(k: int, n: int, e: int, seed=0):
    """Generate H matrix.

    Args:
        k (int): Number of vertices in each hyperedge.
        n (int): Number of nodes.
        e (int): Number of hyperedges.

    Returns:
        The hypergraph and H matrix.
    """
    # genearate a random hypergraph
    dhg.random.set_seed(seed)
    hg = dhg.random.uniform_hypergraph_Gnm(k, n, e)
    nums_v = len(set(np.hstack(hg.e[0])))
    e = compact_vertex_indices(hg.e[0], strat=0)
    hg = dhg.Hypergraph(nums_v, e)
    return hg


def from_hypergraph_to_graph_clique(hypergraph, remove_self_loops=True):
    tmp = dhg.Graph.from_hypergraph_clique(hypergraph)
    edge_list = tmp.e[0]
    edges = []
    loops = []
    for edge in edge_list:
        if edge[0] == edge[1]:
            loops.append(edge)
            continue
        edges.append(edge)
    if not remove_self_loops:
        edges = edges + loops
    g = Graph(num_v=hypergraph.num_v, e_list=edges)
    logger.info(f"INFO: {len(loops)} loops found")
    return g


def get_H(hg):
    """Generate H matrix.
    Args:
        hg: The hypergraph.
    """
    num_vertices = hg.num_v
    num_hyperedges = hg.num_e
    H = torch.zeros((num_hyperedges, num_vertices), dtype=torch.float)
    for i, hyperedge in enumerate(hg.e[0]):
        for vertex in hyperedge:
            H[i, vertex] = 1
    return H.T


def generate_C(e: int, seed=0):
    """Generate C matrix.
    Args:
        e (int): number of hyperedges.
    """
    torch.manual_seed(seed)
    return 2 * torch.rand(1, e) - 1.0


# pubo solution
def solve_pubo_with_pyqubo(num_nodes, num_hyperedges, H, C, method="tabu", seed=0):
    """
    Solve a PUBO problem defined by a hypergraph.

    Args:
        num_nodes (int): Number of nodes.
        num_hyperedges (int): Number of hyperedges.
        H (tensor): H matrix.
        C (tensor): C matrix.
        method (str): Method to solve the problem ('tabu' or 'annealing').
        seed (int): Random seed for reproducibility.

    Returns:
        solution (dict): Optimal solution for the binary variables.
        energy (float): Energy value of the optimal solution.
    """

    # Convert H and C to numpy arrays for compatibility
    H_np = H.numpy()
    C_np = C.numpy()

    # Create binary variables for the PUBO problem
    x = Array.create("x", shape=num_nodes, vartype="BINARY")

    # Build the PUBO objective function
    Q = 0
    for j in range(num_hyperedges):
        term = 1
        for i in range(num_nodes):
            if H_np[i, j] == 1:
                term *= x[i]  # Multiply variables in the hyperedge
        Q += C_np[0, j] * term  # Add the term weighted by C

    # Compile the PUBO model
    model = Q.compile()
    qubo, offset = model.to_qubo()

    # Choose the solver based on the method
    if method == "tabu":
        sampler = SimulatedAnnealingSampler()  # Use Tabu search
    else:
        sampler = dimod.SimulatedAnnealingSampler()  # Use simulated annealing

    # Sample the QUBO problem to find solutions
    response = sampler.sample_qubo(qubo, num_reads=2000)
    # Extract the optimal solution and its loss
    solution = response.first.sample
    loss = response.first.energy

    return solution, loss


def solve_pubo_with_scip(num_nodes, num_hyperedges, H, C, time_limit=3600, seed=0):
    """
    Solve the PUBO problem using SCIP.

    Args:
        num_nodes (int): Number of nodes.
        num_hyperedges (int): Number of hyperedges.
        H (tensor): H matrix.
        C (tensor): C matrix.
        time_limit: Time limit. default to 300.
        seed (int): Random seed for reproducibility.

    Returns:
        solution (dict): Optimal solution for the binary variables.
        objective_value (float): Optimal objective value.
    """
    # Convert H and C to numpy arrays
    H_np = H.numpy()
    C_np = C.numpy()

    # Create a SCIP model
    model = Model("PUBO")

    # Add binary variables
    x = {}
    for i in range(num_nodes):
        x[i] = model.addVar(vtype="B", name=f"x_{i}")  # Binary variable

    # Build the PUBO objective function
    Q = 0
    for j in range(num_hyperedges):
        term = 1
        for i in range(num_nodes):
            if H_np[i, j] == 1:
                term *= x[i]  # Multiply variables in the hyperedge
        Q += C_np[0, j] * term  # Add the term weighted by C

    # Set the objective function
    nonlinear.set_nonlinear_objective(model, Q, "minimize")

    # Set time limit
    model.setRealParam("limits/time", time_limit)

    # Solve the problem
    model.optimize()

    # Extract the solution
    solution = {}
    for i in range(num_nodes):
        solution[f"x_{i}"] = int(model.getVal(x[i]))  # Get the value of the binary variable

    # Get the objective value
    loss = model.getObjVal()

    # Print the results
    print("Optimal solution:", solution)
    print("Objective value:", loss)

    return solution, loss


def solve_pubo_with_tabu(num_nodes, num_hyperedges, H, C, max_iter=1000, tabu_tenure=10, seed=0):
    """
    Solve the PUBO problem using Tabu Search algorithm.

    Args:
        num_nodes (int): Number of nodes.
        num_hyperedges (int): Number of hyperedges.
        H (tensor): H matrix of shape (num_nodes, num_hyperedges).
        C (tensor): C matrix of shape (1, num_hyperedges).
        max_iter (int): Maximum number of iterations. Defaults to 1000.
        tabu_tenure (int): Tabu tenure parameter. Defaults to 10.
        seed (int): Random seed. Defaults to 0.

    Returns:
        solution (dict): Best solution found.
        loss (float): Energy of the best solution.
    """
    # Preprocess hyperedge structures
    variables_in_edge = []
    for j in range(num_hyperedges):
        edge_vars = torch.nonzero(H[:, j], as_tuple=True)[0].tolist()
        variables_in_edge.append(edge_vars)

    edges_for_var = [[] for _ in range(num_nodes)]
    for j in range(num_hyperedges):
        for i in variables_in_edge[j]:
            edges_for_var[i].append(j)

    # Initialize random solution
    torch.manual_seed(seed)
    x = torch.randint(0, 2, (num_nodes,)).tolist()
    best_x = x.copy()

    # Initialize active edges and energy
    active_edges = [False] * num_hyperedges
    for j in range(num_hyperedges):
        active = all(x[i] == 1 for i in variables_in_edge[j])
        active_edges[j] = active
    current_energy = sum(C[0, j].item() * active_edges[j] for j in range(num_hyperedges))
    best_energy = current_energy

    # Tabu list (key: variable index, value: expiration iteration)
    tabu_list = dict()

    for iteration in range(max_iter):
        best_delta = float("inf")
        best_move = None

        # Evaluate all possible moves
        for i in range(num_nodes):
            # Skip tabu-active variables (except aspiration)
            if iteration < tabu_list.get(i, -1):
                continue

            # Calculate delta for flipping variable i
            delta = 0
            for j in edges_for_var[i]:
                # Original activation state
                old_active = active_edges[j]

                # Check new activation state
                new_active = True
                for k in variables_in_edge[j]:
                    # Use flipped value for current variable
                    val = 1 - x[i] if k == i else x[k]
                    if val != 1:
                        new_active = False
                        break

                # Accumulate energy change
                delta += C[0, j].item() * (new_active - old_active)

            # Update best move
            if delta < best_delta:
                best_delta = delta
                best_move = i

        # Perform the best move
        if best_move is not None:
            i = best_move
            # Update solution and energy
            x[i] = 1 - x[i]
            current_energy += best_delta

            # Update active edges for affected hyperedges
            for j in edges_for_var[i]:
                active_edges[j] = all(x[k] == 1 for k in variables_in_edge[j])

            # Update tabu list
            tabu_list[i] = iteration + tabu_tenure

            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            break  # No valid moves

    # Convert to dictionary format
    solution = {f"x_{i}": int(best_x[i]) for i in range(num_nodes)}
    return solution, best_energy


def compact_vertex_indices(edge_list: list[tuple], strat: int = 1, strategy: str = "lexicographic") -> list[tuple[int]]:
    """
    Renumber vertices with continuous indices starting from `strat`, supporting two mapping strategies.

    Args:
        edge_list: Original list of hyperedges containing vertices.
        strat: Starting index for the new indices (default=1).
        strategy: Mapping strategy, 'appearance' (default) or 'lexicographic'.

    Returns:
        List of reindexed hyperedges with continuous indices.

    Examples:
        >>> edge_list = [[1, 3, 7], [2, 11], [3, 666, 987]]
        >>> compact_vertex_indices(edge_list, strat=1, strategy='appearance')
        [(1, 2, 3), (4, 5), (2, 6, 7)]
        >>> compact_vertex_indices(edge_list, strat=1, strategy='lexicographic')
        [(1, 3, 4), (2, 5), (3, 6, 7)]
    """
    if strategy == "lexicographic":
        vertices = sorted({v for hyperedge in edge_list for v in hyperedge})
        mapping = {v: i + strat for i, v in enumerate(vertices)}
    else:
        from collections import defaultdict

        mapping = defaultdict(lambda: strat + len(mapping))
        for hyperedge in edge_list:
            for v in hyperedge:
                _ = mapping[v]

    return [tuple(mapping[v] for v in hyperedge) for hyperedge in edge_list]


def get_dnn_solution(outs: torch.Tensor):
    """
    Extract the task solution provided by the model from the final output,
    which may serve as the initial solution for hot-starting traditional methods.

    Args:
        outs (outs: torch.Tensor): DNN model output, shape [V, 1]

    Returns:
        out (Dict): Example: {0: 1, 1: 0, 2: 1, 3:1, ...}
    """
    detached = outs.detach().cpu()
    return {idx: sample for idx, sample in enumerate((detached > 0.5).int().flatten().numpy())}


def from_hypergraph_to_graph_clique(hypergraph, remove_self_loops=True):
    tmp = dhg.Graph.from_hypergraph_clique(hypergraph)
    edge_list = tmp.e[0]
    edges = []
    loops = []
    for edge in edge_list:
        if edge[0] == edge[1]:
            loops.append(edge)
            continue
        edges.append(edge)
    if not remove_self_loops:
        edges = edges + loops
    g = Graph(num_v=hypergraph.num_v, e_list=edges)
    logger.info(f"INFO: {len(loops)} loops found")
    return g


def from_file_to_hypergraph(file_path: str, reset_vertex_index: bool = False) -> Hypergraph:
    """Read a hypergraph from a file with support for string vertex names.

    Args:
        file_path: Path to the hypergraph file
        reset_vertex_index: Whether to reindex vertices starting from 0

    Returns:
        Hypergraph with parsed vertices and hyperedges
    """
    with open(file_path, "r") as file:
        # Read and preprocess lines
        lines = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

        # Skip optional first line with two numbers
        if lines and len(lines[0].split()) == 2:
            lines = lines[1:]

        # Parse edges as strings
        edges = [line.split() for line in lines if len(line.split()) >= 2]

        # Create sorted vertex list with numeric
        def sort_key(v):
            try:
                return (0, int(v))
            except ValueError:
                return (1, v)

        all_vertices = sorted({v for edge in edges for v in edge}, key=sort_key)
        # all_vertices = {v for edge in edges for v in edge}
        # Create mapping if reindexing needed
        vertex_mapping = {old: idx for idx, old in enumerate(all_vertices)}

        # Process edges based on reindex flag
        processed_edges = []
        for edge in edges:
            if reset_vertex_index:
                processed_edges.append([vertex_mapping[v] for v in edge])
            else:
                processed_edges.append(edge)

        # Create hypergraph instance
        return Hypergraph(num_v=len(all_vertices), e_list=processed_edges)


def from_pickle_to_hypergraph(dataset: str) -> Any:
    # fmt: off
    
    r"""
    
    Read a hypergraph from a pickle file.
    
    ---
    Args:
        ``file_path``(`str`):  The path to the pickle file containing the hypergraph.

    Returns:
        HG(`Hypergraph`):  The hypergraph read from the pickle file.  
    
    Example:
    
    .. code-block:: python
        hg = from_pickle_to_hypergraph("data/test_hypergraph")
    """
    # fmt: on
    from dhg import Hypergraph

    data_path = os.path.join(dataset)

    with open(os.path.join(data_path, f"H.pickle"), "rb") as f:
        H = pkl.load(f)
    l: dict[int, list] = {}
    for i, j in zip(H[0], H[1]):
        i, j = i.item(), j.item()
        if l.get(j):
            l[j].append(i)
        else:
            l[j] = [i]
    sorted_l = {k: v for k, v in sorted(l.items(), key=lambda item: item[0])}
    num_v = H[0].max().item() + 1
    e_list = list(sorted_l.values())
    return Hypergraph(num_v, e_list, merge_op="mean")
