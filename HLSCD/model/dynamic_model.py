import torch
from torch import nn
from torch.func import jacrev, vmap

class LatentCausalDynamics(nn.Module):
    def __init__(self, n_clusters: int,hidden_dim:int,diffusion: float = 0.01) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.flow = nn.Sequential(nn.Linear(n_clusters, hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,n_clusters))
        self.dt = nn.Parameter(torch.tensor(0.75))
        self.register_buffer("diffusion", torch.tensor(float(diffusion)))

    def vector_field(self, z_state: torch.Tensor) -> torch.Tensor:
        nonlinear_effect = self.flow(z_state)
        return nonlinear_effect

    def flow_jacobian(self, z_series: torch.Tensor) -> torch.Tensor:
        # Local Jacobian d self.flow(z) / d z for every batch/time latent state.
        n_slots = z_series.size(-1)
        flat_z = z_series.reshape(-1, n_slots)
        grad_enabled = torch.is_grad_enabled()
        def flow_single(z_state: torch.Tensor) -> torch.Tensor:
            return self.flow(z_state)

        with torch.enable_grad():
            jac = vmap(jacrev(flow_single))(flat_z)
        if not grad_enabled:
            jac = jac.detach()
        return jac.reshape(*z_series.shape[:-1], n_slots, n_slots)

    def forward(self, z_series: torch.Tensor):
        z_last = z_series
        latent_graph = self.flow_jacobian(z_series).mean(dim=1)
        step = self.dt.clamp(0.5, 1.5)
        drift = self.vector_field(z_last)
        if self.training:
            noise = self.diffusion * torch.sqrt(step) * torch.randn_like(z_last)
        else:
            noise = torch.zeros_like(z_last)
        z_next = z_last + step * drift + noise
        return z_next, latent_graph
