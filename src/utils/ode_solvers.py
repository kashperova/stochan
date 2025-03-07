from typing import Callable

from torch import Tensor


class ODESolver:
    def solve(self, z0: Tensor, t0: Tensor, t1: Tensor, func: Callable):
        raise NotImplementedError

    @staticmethod
    def get_steps(t0: Tensor, t1: Tensor):
        h_max = 0.01
        n_steps = int(((t1 - t0).abs() / h_max).ceil().item())
        h = (t1 - t0) / n_steps
        return n_steps, h


class EulerSolver(ODESolver):
    def solve(self, z: Tensor, t0: Tensor, t1: Tensor, func: Callable):
        n_steps, h = self.get_steps(t0, t1)
        t = t0
        for _ in range(n_steps):
            z = z + h * func(z, t)
            t += h

        return z


class RungeKuttaSolver(ODESolver):
    def solve(self, z: Tensor, t0: Tensor, t1: Tensor, func: Callable):
        n_steps, h = self.get_steps(t0, t1)
        t = t0
        for _ in range(n_steps):
            k1 = func(z, t)
            k2 = func(z + 0.5 * h * k1, t + 0.5 * h)
            k3 = func(z + 0.5 * h * k2, t + 0.5 * h)
            k4 = func(z + h * k3, t + h)
            z = z + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += h
