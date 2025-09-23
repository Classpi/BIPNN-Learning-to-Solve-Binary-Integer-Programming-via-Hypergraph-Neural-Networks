import os
import time
import random
import logging

from enum import Enum
from pathlib import Path
from inspect import signature
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

import numpy
import torch
import torch_geometric
from torch import nn

from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING

_DEVICE: Optional[torch.device] = None
_SEED = None


class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.WARNING: "\033[31m",  # RED
        logging.ERROR: "\033[31;1m",  # HIGHRED
        logging.INFO: "\033[32m",  # GREEN
        logging.DEBUG: "\033[34m",  # BLUE
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = ColorFormatter(fmt="[%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        raise RuntimeError("Device not initialized. Call init() first.")
    return _DEVICE


def get_current_seed() -> int:
    global _SEED
    if _SEED is None:
        raise RuntimeError("Seed not initialized. Call init() first.")
    return _SEED


def init(
    device: Optional[torch.device] = None,
    cuda_index: int = 0,
    verbose: bool = True,
    reproducibility: bool = False,
    seed=42,
):
    os.environ["PYTHONHASHSEED"] = str(seed)

    global _DEVICE, _SEED
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device is not None:
        _DEVICE = device
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > cuda_index:
            if reproducibility:
                logger.warning(
                    "You have enabled the reproducibility feature, which uses a deterministic non-optimized algorithm, greatly affecting the running efficiency"
                )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True)
                torch.set_deterministic_debug_mode("warn")
            _DEVICE = torch.device(f"cuda:{cuda_index}")
            torch.cuda.set_device(cuda_index)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch_geometric.seed_everything(seed)

            if verbose:
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(cuda_index)} (Index: {cuda_index})")
        else:
            _DEVICE = torch.device("cpu")
            if verbose:
                logger.warning("CUDA device not available. Using CPU.")


class BaseSCIPSolver(ABC):
    def __init__(self, edge_list, pre_solve=True):
        self.edge_list = edge_list
        self.V = list(sorted({v for e in edge_list for v in e}))
        self.model = Model("Problem")
        self.solution_history = []
        self.start_time = None
        self.pre_solve = pre_solve

    @abstractmethod
    def _add_variables(self):
        pass

    @abstractmethod
    def _add_constraints(self):
        pass

    @abstractmethod
    def _set_objective(self):
        pass

    @abstractmethod
    def _get_solution_metrics(self, sol):
        pass

    @abstractmethod
    def _extract_solution(self, sol):
        pass

    def _setup_model(self, time_limit, sol_limit):
        self.model.setRealParam("limits/time", time_limit)
        logger.debug(f"set scip time limit: {time_limit}")
        if sol_limit is not None:
            self.model.setRealParam("limits/primal", sol_limit)
            logger.debug(f"set scip solution limit: {sol_limit}")
        self.model.setPresolve(SCIP_PARAMSETTING.OFF) if not self.pre_solve else None

    class SolutionTracker(Eventhdlr):
        def __init__(self, solver):
            super().__init__()
            self.solver = solver

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            elapsed_time = time.time() - self.solver.start_time
            current_sol = self.model.getBestSol()
            metrics = self.solver._get_solution_metrics(current_sol)
            self.solver.solution_history.append((elapsed_time, metrics))
            print(f"[{elapsed_time:.2f}s] Current solution: {metrics}")

    def solve(self, time_limit=3600, sol_limit=None):
        self.start_time = time.time()
        self.solution_history = []
        self._setup_model(time_limit, sol_limit)
        self._add_variables()
        self._add_constraints()
        self._set_objective()

        tracker = self.SolutionTracker(self)
        self.model.includeEventhdlr(tracker, "SolutionTracker", "Tracks solutions")

        self.model.optimize()

        if self.model.getStatus() == "optimal":
            sol = self.model.getBestSol()
            final_time = time.time() - self.start_time
            metrics = self._get_solution_metrics(sol)
            print(f"\n[{final_time:.2f}s] Optimal solution found. Metrics: {metrics}")
            return self._extract_solution(sol)
        else:
            return self.model.getBestSol()
