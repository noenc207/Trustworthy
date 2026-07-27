"""Optimizer framework for the training engine."""

import collections
from typing import Any

import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    def __init__(self, params: Any, base_optimizer: Any, rho: float = 0.05, adaptive: bool = False, **kwargs: Any):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

class Lookahead(Optimizer):
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = collections.defaultdict(dict)

        for group in self.param_groups:
            group["k_counter"] = 0

        self.fast_state = self.optimizer.state

    def update(self, group: dict) -> None:
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure: Any = None) -> Any:
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["k_counter"] == 0:
                for fast in group["params"]:
                    param_state = self.state[fast]
                    if "slow_param" not in param_state:
                        param_state["slow_param"] = torch.zeros_like(fast.data)
                        param_state["slow_param"].copy_(fast.data)
            group["k_counter"] += 1
            if group["k_counter"] >= self.k:
                group["k_counter"] = 0
                self.update(group)
        return loss

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

class OptimizerFactory:
    @staticmethod
    def create(name: str, params: Any, **kwargs: Any) -> Optimizer:
        name = name.lower()
        if name == 'sgd':
            return torch.optim.SGD(params, **kwargs)
        elif name == 'adam':
            return torch.optim.Adam(params, **kwargs)
        elif name == 'adamw':
            return torch.optim.AdamW(params, **kwargs)
        elif name == 'rmsprop':
            return torch.optim.RMSprop(params, **kwargs)
        elif name == 'sam':
            base_optimizer = kwargs.pop('base_optimizer', torch.optim.SGD)
            return SAM(params, base_optimizer=base_optimizer, **kwargs)
        elif name == 'lookahead_adam':
            k = kwargs.pop('k', 5)
            alpha = kwargs.pop('alpha', 0.5)
            base_opt = torch.optim.Adam(params, **kwargs)
            return Lookahead(base_opt, k=k, alpha=alpha)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
