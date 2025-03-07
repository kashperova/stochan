from typing import Callable

import torch.autograd
from torch import nn, Tensor


class ODEFunc(nn.Module):
    def __init__(self, solver: Callable):
        super().__init__()
        self.solver = solver

    def forward_with_a(self, z: Tensor, t: Tensor, a: Tensor):
        # compute f and a df/dz, a df/dp, a df/dt
        output = self.forward(z, t)

        a_dfdz, a_dfdt, *a_dfdp = torch.autograd.grad(
            outputs=output,
            inputs=(z, t) + tuple(self.parameters()),
            grad_outputs=(a),
            allow_unused=True,
            retain_graph=True
        )
        # torch autograd accumulates gradients across the entire batch
        # so we need to expand back gradients per sample
        batch_size = z.shape[0]

        if a_dfdt is not None:
            a_dfdt = a_dfdt.expand(batch_size, -1) / batch_size

        if a_dfdp is not None:
            a_dfdp = torch.cat([p_grad.flatten() for p_grad in a_dfdp]).unsqueeze(0)
            a_dfdp = a_dfdp.expand(batch_size, -1) / batch_size

        return output, a_dfdz, a_dfdt, a_dfdp

    @property
    def params_vector(self) -> Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])
