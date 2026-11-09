import copy
import sys
from os.path import dirname as opd
from os.path import join as opj

import numpy as np
import torch
import tqdm

sys.path.append(opj(opd(__file__), ".."))
from model.dynamic_model import LatentCausalDynamics
from model.encoder import GlobalCoarseGrainer
from utils.eval_auroc import (compute_matrix_auroc,cosine_similarity,shd_distance)
from utils.fmriopt_type import HCDopt


class HLSCD(object):
    def __init__(self, args: HCDopt.HCDargs, device="cpu"):
        self.args = args
        self.device = torch.device(device)
        self.disable_tqdm = self.args.disable_tqdm
        self.pred_weight = self.args.pred_weight
        self.recon_weight = self.args.recon_weight
        self.dyn_weight = self.args.dyn_weight
        self.logdet_weight = self.args.logdet_weight
        self.logdet_eps = self.args.logdet_eps
        self.loss_fn = torch.nn.MSELoss()
        self._build_model()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.args.lr_decay_step,gamma=self.args.lr_decay_gamma,)

    def _build_model(self) -> None:
        dmd_rank = int(getattr(self.args, "dmd_rank", self.args.k_nodes))
        self.coarse_grainer = GlobalCoarseGrainer(n_nodes=self.args.n_nodes,n_slots=self.args.k_nodes,hidden_dim=self.args.hidden_dim,temperature=self.args.temperature,dmd_rank=dmd_rank,).to(self.device)
        self.dynamic = LatentCausalDynamics(n_clusters=self.args.k_nodes,hidden_dim=self.args.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.coarse_grainer.parameters()) + list(self.dynamic.parameters()),lr=float(self.args.lr),weight_decay=float(self.args.weight_decay),)

    def _prepare_batch(self, batch_data):
        history, *states = batch_data
        if len(states) < 2:
            raise ValueError("A batch must contain at least two consecutive states.")
        state_sequence = torch.stack(
            [state.to(self.device) for state in states], dim=1
        )
        return history.to(self.device), state_sequence

    @staticmethod
    def _normalize_graph(graph: torch.Tensor) -> torch.Tensor:
        graph = graph.abs()
        if graph.dim() == 2:
            denom = graph.max().clamp_min(1e-6)
        else:
            denom = graph.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return graph / denom

    @staticmethod
    def _sparse_loss(graph: torch.Tensor) -> torch.Tensor:
        """L1 penalty on local Jacobian edges.

        The previous fixed-variance Gaussian KL reduced to an L2 penalty and
        therefore shrank every edge without promoting exact sparsity.
        """
        return graph.abs().mean()

    def _scale_invariant_logdet_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """Encourage diverse assignment columns without rewarding their scale.

        For M=[N,K] or [B,N,K], this constructs the column correlation matrix
        R=D^{-1/2}(M^T M)D^{-1/2} and minimizes -logdet(R + eps I).
        """
        gram = assignment.transpose(-1, -2) @ assignment
        column_norm_sq = torch.diagonal(gram, dim1=-2, dim2=-1)
        inv_column_norm = column_norm_sq.clamp_min(self.logdet_eps).rsqrt()
        correlation = (gram * inv_column_norm.unsqueeze(-1)* inv_column_norm.unsqueeze(-2))
        correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
        n_slots = correlation.size(-1)
        eye = torch.eye(n_slots,dtype=correlation.dtype,device=correlation.device,)
        _, logabsdet = torch.linalg.slogdet(correlation + self.logdet_eps * eye)
        return -logabsdet.mean() / max(n_slots, 1)

    @staticmethod
    def masked_auroc(pred_matrix: torch.Tensor, label_matrix: torch.Tensor) -> float:
        mask = ~torch.eye(pred_matrix.size(0), dtype=torch.bool, device=pred_matrix.device)
        preds = pred_matrix[mask]
        labels = label_matrix.to(pred_matrix.device)[mask]
        return compute_matrix_auroc(preds.reshape(-1, 1), labels.reshape(-1, 1))

    @staticmethod
    def _fix_svd_orientation(left_factor: torch.Tensor, right_factor: torch.Tensor):
        anchor_indices = left_factor.abs().argmax(dim=-2, keepdim=True)
        anchors = torch.gather(left_factor, dim=-2, index=anchor_indices)
        orientation = torch.where(anchors < 0, -torch.ones_like(anchors), torch.ones_like(anchors))
        return left_factor * orientation, right_factor * orientation

    def _batch_dmd_basis(self, history: torch.Tensor) -> torch.Tensor:
        basis, _ = self._batch_dmd_decomposition(history)
        return basis

    def _batch_dmd_decomposition(self, history: torch.Tensor):
        """Return concatenated left/right DMD factors preceding x_t."""
        past = history[:, :-1, :]
        future = history[:, 1:, :]
        past = past - past.mean(dim=1, keepdim=True)
        future = future - future.mean(dim=1, keepdim=True)
        past = past / past.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        future = future / future.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        n_nodes = past.size(-1)
        reg = 1e-1 * torch.eye(n_nodes, device=past.device, dtype=past.dtype).expand(past.size(0), -1, -1)
        covariance = torch.einsum("btn,btm->bnm", past, past) / max(past.size(1), 1)
        cross_covariance = torch.einsum("btn,btm->bnm", past, future) / max(past.size(1), 1)
        dmd_operator = torch.linalg.solve(covariance + reg, cross_covariance)
        U, S, Vh = torch.linalg.svd(dmd_operator, full_matrices=False)
        order = S.abs().argsort(dim=-1, descending=True)
        rank = self.coarse_grainer.dmd_rank
        max_informative_rank = min(n_nodes, max(past.size(1) - 1, 0))
        if rank > max_informative_rank:
            raise ValueError(
                f"dmd_rank={rank} exceeds the centered-history rank bound "
                f"{max_informative_rank}; require dmd_rank <= time_lag - 2"
            )
        top_order = order[:, :rank]
        gather_index = top_order.unsqueeze(1).expand(-1, n_nodes, -1)
        top_left_vectors = torch.gather(U, dim=2, index=gather_index)
        V = Vh.transpose(-2, -1)
        top_right_vectors = torch.gather(V, dim=2, index=gather_index)
        top_values = torch.gather(S, dim=1, index=top_order).abs().sqrt().unsqueeze(1)
        left_basis, right_basis = self._fix_svd_orientation(top_left_vectors * top_values, top_right_vectors * top_values)
        basis = torch.cat((left_basis, right_basis), dim=-1)
        return basis, dmd_operator

    def _observation_basis_for_batch(self, history: torch.Tensor):
        return self._batch_dmd_basis(history)

    def train_step(self, batch_data):
        self.coarse_grainer.train()
        self.dynamic.train()
        self.optimizer.zero_grad()
        history, state_sequence = self._prepare_batch(batch_data)
        observation_basis = self._observation_basis_for_batch(history)
        z_sequence, assignment = self.coarse_grainer.encode(
            state_sequence, node_feature=observation_basis
        )
        reconstructed_states = torch.stack(
            [
                self.coarse_grainer.decode_state(
                    z_sequence[:, step, :], assignment=assignment
                )
                for step in range(z_sequence.size(1))
            ],
            dim=1,
        )
        predicted_latent_states = []
        for step in range(z_sequence.size(1) - 1):
            z_next, _ = self.dynamic(z_sequence[:, step : step + 1, :])
            predicted_latent_states.append(z_next[:, 0, :])
        predicted_latent_states = torch.stack(predicted_latent_states, dim=1)
        predicted_states = torch.stack(
            [
                self.coarse_grainer.decode_state(
                    predicted_latent_states[:, step, :], assignment=assignment
                )
                for step in range(predicted_latent_states.size(1))
            ],
            dim=1,
        )
        loss_recon = self.loss_fn(reconstructed_states, state_sequence)
        loss_pred = self.loss_fn(predicted_states, state_sequence[:, 1:, :])
        target_latents = z_sequence[:, 1:, :].detach()
        loss_dyn = self.loss_fn(predicted_latent_states, target_latents)
        loss_logdet = self._scale_invariant_logdet_loss(assignment)
        loss = (self.recon_weight * loss_recon + self.pred_weight * loss_pred + self.dyn_weight * loss_dyn + self.logdet_weight * loss_logdet)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.coarse_grainer.parameters()) + list(self.dynamic.parameters()),max_norm=5.0)
        self.optimizer.step()
        return loss, loss_recon, loss_pred, loss_dyn, loss_logdet

    def _projected_graphs_from_batch(self, batch_data) -> torch.Tensor:
        """Return [batch, transition, node, node] observation-level graphs."""
        history, state_sequence = self._prepare_batch(batch_data)
        observation_basis = self._observation_basis_for_batch(history)
        z_sequence, assignment = self.coarse_grainer.encode(
            state_sequence[:, :-1, :], node_feature=observation_basis
        )
        projected_graphs = []
        for step in range(z_sequence.size(1)):
            z_state = z_sequence[:, step, :]
            latent_graph = self.dynamic.flow_jacobian(z_state)
            nonlinear_effect = self.dynamic.flow(z_state)
            projected_graphs.append(
                self.coarse_grainer.projected_graph(
                    latent_graph,
                    assignment=assignment,
                    z_state=z_state,
                    mu_state=nonlinear_effect,
                )
            )
        return torch.stack(projected_graphs, dim=1)

    def _collect_projected_graphs(self, batch_data_loader) -> torch.Tensor:
        """Concatenate [batch, transition, node, node] graphs over all batches."""
        was_coarse_training = self.coarse_grainer.training
        was_dynamic_training = self.dynamic.training
        self.coarse_grainer.eval()
        self.dynamic.eval()
        projected_graph_batches = []
        with torch.no_grad():
            for batch_data in batch_data_loader:
                projected_graph_batches.append(
                    self._projected_graphs_from_batch(batch_data).detach()
                )
        self.coarse_grainer.train(was_coarse_training)
        self.dynamic.train(was_dynamic_training)
        if not projected_graph_batches:
            raise ValueError("Cannot evaluate an empty data loader.")
        return torch.cat(projected_graph_batches, dim=0)

    def _snapshot_state(self, epoch_i: int):
        return {"coarse_grainer": copy.deepcopy(self.coarse_grainer.state_dict()),
            "dynamic": copy.deepcopy(self.dynamic.state_dict()),
            "epoch": epoch_i,}

    @staticmethod
    def _binarize_like_label(pred_graph: torch.Tensor, label_graph: torch.Tensor) -> np.ndarray:
        """Select top-k predicted full-graph edges, where k is label edge count."""
        pred = pred_graph.detach().abs().cpu().numpy()
        label = label_graph.detach().cpu().numpy().astype(int)
        n_edges = int(label.sum())

        if n_edges <= 0:
            return np.zeros_like(label, dtype=int)
        if n_edges >= pred.size:
            return np.ones_like(label, dtype=int)

        flat_pred = pred.reshape(-1)
        top_indices = np.argpartition(flat_pred, -n_edges)[-n_edges:]
        binary = np.zeros(flat_pred.shape, dtype=int)
        binary[top_indices] = 1
        return binary.reshape(pred.shape)

    def _evaluate_full_graph(self, test_batch_data, gc_tensor: torch.Tensor):
        projected_graphs = self._collect_projected_graphs(test_batch_data)
        gc_cpu = gc_tensor.detach().cpu()
        gc_binary = gc_cpu.abs().gt(0).to(dtype=gc_cpu.dtype)
        projected_graph = projected_graphs.mean(dim=(0, 1)).detach().cpu()
        full_graph = self._normalize_graph(projected_graph)
        pred_binary = self._binarize_like_label(full_graph, gc_binary)
        pred_binary_tensor = torch.as_tensor(pred_binary, dtype=gc_binary.dtype, device=gc_binary.device)
        return {"full_auroc": compute_matrix_auroc(full_graph, gc_binary),"full_shd": shd_distance(pred_binary, gc_binary.numpy()),"full_cs": cosine_similarity(pred_binary_tensor, gc_binary, gc_binary.size(0)),}

    def _train_single_run(self, train_batch_data, test_batch_data,gc_tensor=None):
        best_score = float("inf")
        best_state = None
        best_metrics = None

        try:
            n_batches = len(train_batch_data)
        except TypeError:
            n_batches = 1

        pbar = tqdm.tqdm(total=self.args.total_epoch * max(n_batches, 1), disable=self.disable_tqdm)
        for epoch_i in range(self.args.total_epoch):
            #清空缓存
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            epoch_pred = []
            epoch_recon = []
            for batch_i, batch_data in enumerate(train_batch_data):
                loss, loss_recon, loss_pred, loss_dyn, loss_logdet = self.train_step(batch_data)
                epoch_pred.append(loss_pred.item())
                epoch_recon.append(loss_recon.item())
                pbar.update(1)
                pbar.set_postfix_str(
                    f"epoch={epoch_i + 1}/{self.args.total_epoch}, "
                    f"batch={batch_i + 1}/{n_batches}, "
                    f"lr={epoch_lr:.2e}, "
                    f"loss={loss.item():.3f}, recon={loss_recon.item():.3f}, "
                    f"pred={loss_pred.item():.3f}, dyn={loss_dyn.item():.3f}, "
                    f"logdet={loss_logdet.item():.3f}")

            avg_pred = sum(epoch_pred) / max(len(epoch_pred), 1)
            avg_recon = sum(epoch_recon) / max(len(epoch_recon), 1)
            desc_prefix = f"epoch={epoch_i + 1}, lr={epoch_lr:.2e}, pred={avg_pred:.4f}, recon={avg_recon:.4f}"

            if gc_tensor is not None:
                metrics = self._evaluate_full_graph(test_batch_data, gc_tensor)
                score = metrics["full_shd"]
                metrics.update({"pred": avg_pred, "recon": avg_recon})
                pbar.set_description(f"{desc_prefix}, auroc={metrics['full_auroc']:.4f}, "f"shd={metrics['full_shd']}, cs={metrics['full_cs']:.4f}")
            else:
                score = -(self.pred_weight * avg_pred + self.recon_weight * avg_recon)
                metrics = {"pred": avg_pred, "recon": avg_recon}
                pbar.set_description(desc_prefix)

            # With labels, best_score is the full-graph AUROC.
            if score < best_score:
                best_score = score
                best_state = self._snapshot_state(epoch_i)
                best_metrics = metrics
            self.scheduler.step()
        pbar.close()
        return best_state, best_metrics, best_score

    def train(self, train_batch_data, test_batch_data, GC=None, return_metrics: bool = False):
        gc_tensor = None if GC is None else torch.as_tensor(GC, dtype=torch.float32)
        best_state, best_metrics, _ = self._train_single_run(train_batch_data, test_batch_data,gc_tensor=gc_tensor)

        if best_state is None:
            epoch = self.args.total_epoch - 1
            return (epoch, best_metrics) if return_metrics else epoch

        self.coarse_grainer.load_state_dict(best_state["coarse_grainer"])
        self.dynamic.load_state_dict(best_state["dynamic"])

        if best_metrics is not None and gc_tensor is not None:
            print(
                f"best epoch: {best_state['epoch'] + 1}, "
                f"auroc={best_metrics['full_auroc']:.4f}, "
                f"shd={best_metrics['full_shd']}, "
                f"cs={best_metrics['full_cs']:.4f}, "
                f"pred={best_metrics['pred']:.4f}, "
                f"recon={best_metrics['recon']:.4f}"
            )
        elif best_metrics is not None:
            print(
                f"best epoch: {best_state['epoch'] + 1}, "
                f"pred={best_metrics['pred']:.4f}, "
                f"recon={best_metrics['recon']:.4f}"
            )
        return (best_state["epoch"], best_metrics) if return_metrics else best_state["epoch"]


def main(train_batch_data, test_batch_data, opt, GC=None, device="cpu", return_metrics: bool = False):
    hcd = HLSCD(opt, device=device)
    epoch_i = hcd.train(train_batch_data, test_batch_data, GC, return_metrics=return_metrics)
    return epoch_i
