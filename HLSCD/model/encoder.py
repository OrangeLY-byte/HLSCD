import torch
from torch import nn
import torch.nn.functional as F
from torch.func import jacfwd, jacrev, vmap
from typing import Optional

class GlobalCoarseGrainer(nn.Module):
    def __init__(self,n_nodes: int,n_slots: int,hidden_dim: int,temperature: float,dmd_rank: Optional[int] = None,) -> None:
        super().__init__()
        dmd_rank = n_slots if dmd_rank is None else int(dmd_rank)
        if dmd_rank < 1:
            raise ValueError("dmd_rank must be positive")
        self.n_nodes = n_nodes
        self.n_slots = n_slots
        self.dmd_rank = dmd_rank
        self.temperature = temperature
        self.node_feature_dim = 2 * dmd_rank
        self.factor_norm = nn.LayerNorm(self.node_feature_dim)
        self.factor_mixer = nn.Linear(self.node_feature_dim, n_slots)
        self.encode_mlp = nn.Sequential(nn.Linear(n_slots, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_slots))
        self.decode_mlp = nn.Sequential(nn.Linear(n_nodes, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_nodes))

    def _assignment_from_node_feature(self, node_feature: torch.Tensor) -> torch.Tensor:
        logits = self.factor_mixer(node_feature)
        return F.softmax(logits / self.temperature, dim=-1)

    def assignment(self, node_feature: torch.Tensor) -> torch.Tensor:
        node_feature = self.factor_norm(node_feature.to(device=self.factor_mixer.weight.device,dtype=self.factor_mixer.weight.dtype,))
        return self._assignment_from_node_feature(node_feature)

    def encode(self, x_window: torch.Tensor, node_feature: torch.Tensor):
        # x_window: [B, T, N]
        assignment = self.assignment(node_feature=node_feature)
        encode_assignment = assignment / (assignment.sum(dim=-2, keepdim=True) + 1e-8)
        weighted_cluster_state = torch.einsum("btn,bnk->btk", x_window, encode_assignment)
        z_series = self.encode_mlp(weighted_cluster_state)
        return z_series, assignment

    def state_x_hat(self, z_state: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        weighted_node_state = torch.einsum("bk,bnk->bn", z_state, assignment)
        return weighted_node_state

    def decode_state(self, z_state: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        x_hat = self.state_x_hat(z_state, assignment=assignment)
        return self.decode_mlp(x_hat)

    def decoder_mlp_derivatives(self, x_hat: torch.Tensor):
        # Per-output derivatives of decoder_mlp(x_hat) with respect to x_hat.
        feature_dim = x_hat.size(-1)
        flat_x = x_hat.reshape(-1, feature_dim)
        grad_enabled = torch.is_grad_enabled()

        def decoder_single(x_state: torch.Tensor) -> torch.Tensor:
            return self.decode_mlp(x_state)

        with torch.enable_grad():
            jac = vmap(jacrev(decoder_single))(flat_x)
            hess = vmap(jacrev(jacrev(decoder_single)))(flat_x)
        if not grad_enabled:
            jac = jac.detach()
            hess = hess.detach()
        output_dim = self.n_nodes
        jac = jac.reshape(*x_hat.shape[:-1], output_dim, feature_dim)
        hess = hess.reshape(*x_hat.shape[:-1], output_dim, feature_dim, feature_dim
        )
        return jac, hess

    def latent_observation_derivatives(self,z_state: torch.Tensor,assignment: torch.Tensor,):
        """Differentiate decode_mlp(Sz) directly in the low-dimensional z space."""

        def observation_single(z_single: torch.Tensor,assignment_single: torch.Tensor,) -> torch.Tensor:
            decoder_input = assignment_single @ z_single
            return self.decode_mlp(decoder_input)

        observation_jacobian = jacrev(observation_single, argnums=0)

        def derivatives_single(z_single: torch.Tensor,assignment_single: torch.Tensor,):
            jacobian = observation_jacobian(z_single, assignment_single)
            hessian = jacfwd(observation_jacobian, argnums=0)(z_single, assignment_single)
            return jacobian, hessian

        return vmap(derivatives_single)(z_state, assignment)

    def _projected_graph_from_derivatives(self,latent_graph: torch.Tensor, z_state: torch.Tensor, assignment: torch.Tensor,mu_state: torch.Tensor, pinv_eps: float, ) -> torch.Tensor:
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            decoder_jacobian, observation_hessian = (self.latent_observation_derivatives(z_state=z_state,assignment=assignment,))
            hessian_drift = torch.einsum("bnij,bj->bni", observation_hessian, mu_state)
            graphs = (decoder_jacobian @ latent_graph + hessian_drift) @ torch.linalg.pinv(decoder_jacobian,rcond=pinv_eps)
        if not grad_enabled:
            graphs = graphs.detach()
        return graphs

    def projected_graph(self, latent_graph: torch.Tensor, assignment: torch.Tensor,z_state: torch.Tensor, mu_state: torch.Tensor, pinv_eps: float = 1e-3, ) -> torch.Tensor:
        return self._projected_graph_from_derivatives(latent_graph=latent_graph,z_state=z_state,assignment=assignment,mu_state=mu_state,pinv_eps=pinv_eps, )
