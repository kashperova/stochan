from torch import Tensor, nn

from models.neural_ode import ODEFunc
from utils.ode_solvers import ODESolver


class Spiral3dODEFunc(ODEFunc):
    def __init__(self, ode_solver: ODESolver, hidden_dim: int = 64):
        super().__init__(solver=ode_solver.solve)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z: Tensor, t: Tensor):
        return self.net(z)  # z shape -> (batch_size, 3)
