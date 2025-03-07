from torch import nn, Tensor

from models.neural_ode import ODEFunc
from models.neural_ode.adjoint import AdjointMethod


class NeuralODE(nn.Module):
    def __init__(self, func: ODEFunc):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEFunc), "function for solving ode should be inherited from ODEFunc class"
        self.func = func

    def forward(self, z0: Tensor, t: Tensor = Tensor([0.0, 1.0]), return_trajectory: bool = False):
        t = t.to(z0)
        z = AdjointMethod.apply(z0, t, self.func.params_vector, self.func)
        if return_trajectory:
            return z

        return z[-1]
