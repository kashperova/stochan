import torch
from torch import Tensor

from models.neural_ode import ODEFunc


def solve_augmented(
    aug_z_step: Tensor,
    t_cur: Tensor,
    t_prev: Tensor,
    func: ODEFunc,
    n_dim: int,
    n_params: int,
    batch_size: int,
    z_shape: tuple,
):
    def augmented_dynamics(aug_z_i: Tensor, t_i: Tensor):
        # aug_z_i.shape -> 2 * n_dim + n_params + 1
        # aug_z_i.shape -> z components + dldz + params grads + t grad
        z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim : 2 * n_dim]

        z_i = z_i.view(batch_size, *z_shape)
        a = a.view(batch_size, *z_shape)

        with torch.set_grad_enabled(True):
            t_i = t_i.detach().requires_grad_(True)
            z_i = z_i.detach().requires_grad_(True)
            out, a_dfdz, a_dfdt, a_dfdp = func.forward_with_a(z_i, t_i, a)

            a_dfdz = a_dfdz.to(z_i) if a_dfdz is not None else torch.zeros(batch_size, *z_shape).to(z_i)
            a_dfdp = a_dfdp.to(z_i) if a_dfdp is not None else torch.zeros(batch_size, n_params).to(z_i)
            a_dfdt = a_dfdt.to(z_i) if a_dfdt is not None else torch.zeros(batch_size, 1).to(z_i)

        return torch.cat((out, -a_dfdz, -a_dfdp, -a_dfdt), dim=1)

    return func.solver(aug_z_step, t_cur, t_prev, augmented_dynamics)


class AdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0: Tensor, t: Tensor, params_vector: Tensor, func: ODEFunc):
        batch, *z_shape = z0.size()
        time_range = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_range, batch, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_range - 1):
                z0 = func.solver(z0, t[i_t], t[i_t + 1], func)
                z[i_t + 1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), params_vector)
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # grad_output - dldz (L - loss function)
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_range, batch_size, *z_shape = z.size()

        n_dim = torch.prod(torch.tensor(z_shape)).item()  # z ∈ R^(n_dim)
        n_params = flat_parameters.size(0)

        dldz = grad_output.view(time_range, batch_size, n_dim)

        with torch.no_grad():
            # previous backward adjoint representations to be updated by direct gradients
            adj_z = torch.zeros(batch_size, n_dim).to(dldz)
            adj_p = torch.zeros(batch_size, n_params).to(dldz)
            # Unlike z and params, gradients for all times must be returned
            adj_t = torch.zeros(time_range, batch_size, 1).to(dldz)

            for i_t in range(time_range - 1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(batch_size, n_dim)

                # direct gradients
                dldz_i = dldz[i_t]
                # chain rule: dldt = dldz * dzdt = dldz * f(z, t)
                dldt_i = torch.bmm(torch.transpose(dldz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # update adjoint representations with direct gradients
                adj_z += dldz_i
                adj_t[i_t] = adj_t[i_t] - dldt_i

                # flatten augmented variable
                aug_z = torch.cat(
                    (z_i.view(batch_size, n_dim), adj_z, torch.zeros(batch_size, n_params).to(z), adj_t[i_t]),
                    dim=-1,
                )

                # solve augmented backwards
                aug_solved = solve_augmented(
                    aug_z_step=aug_z,
                    t_cur=t_i,
                    t_prev=t[i_t - 1],
                    func=func,
                    n_dim=n_dim,
                    n_params=n_params,
                    batch_size=batch_size,
                    z_shape=z_shape
                )
                # unpack solved backwards augmented
                adj_z[:] = aug_solved[:, n_dim : 2 * n_dim]
                adj_p[:] += aug_solved[:, 2 * n_dim : 2 * n_dim + n_params]
                adj_t[i_t - 1] = aug_solved[:, 2 * n_dim + n_params :]

            # update 0 (initial) time adjoint with direct gradients
            # Compute direct gradients
            dldz_0 = dldz[0]
            dldt_0 = torch.bmm(torch.transpose(dldz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # update adjoint representations with direct gradients
            adj_z += dldz_0
            adj_t[0] = adj_t[0] - dldt_0

        return adj_z.view(batch_size, *z_shape), adj_t, adj_p, None
