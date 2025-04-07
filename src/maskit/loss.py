from torch import nn
import torch



class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # log(σ²) for each task; initialized to 0 => σ² = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        # losses: list of per-task scalar loss tensors
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]  # log(σ)
            total_loss += weighted_loss
        return total_loss

class GradNormLoss(nn.Module):
    def __init__(self, num_tasks, alpha=1.5):
        super().__init__()
        self.alpha = alpha
        self.task_weights = nn.Parameter(torch.ones(num_tasks))  # Learnable weights
        self.initial_losses = None  # Will be initialized on first step

    def forward(self, task_losses, shared_layer, model, current_step):
        weighted_losses = self.task_weights.softmax(dim=0) * torch.stack(task_losses)
        total_loss = weighted_losses.sum()

        # Compute gradients of each task loss w.r.t shared layer
        G_norm = []
        for i, loss in enumerate(task_losses):
            model.zero_grad()
            loss.backward(retain_graph=True)

            grad = shared_layer.weight.grad  # could be any shared parameter
            G = torch.norm(self.task_weights[i] * grad)
            G_norm.append(G)

        G_norm = torch.stack(G_norm).detach()

        # Initialize loss ratios on first step
        if self.initial_losses is None:
            self.initial_losses = torch.stack(task_losses).detach()

        loss_ratios = torch.stack(task_losses).detach() / self.initial_losses
        inverse_train_rates = loss_ratios / loss_ratios.mean()

        target_grad = G_norm.mean() * (inverse_train_rates ** self.alpha)
        grad_norm_loss = nn.functional.l1_loss(G_norm, target_grad.detach())

        return total_loss, grad_norm_loss, self.task_weights.softmax(dim=0).detach()

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DynamicWeightAveragingLoss(nn.Module):
    def __init__(self, num_tasks, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("loss_history", torch.ones(num_tasks, 2))  # [t-2, t-1]
        self.register_buffer("task_weights", torch.ones(num_tasks))

    def update_weights(self, current_losses):
        # Previous losses for ratio computation
        prev_losses = self.loss_history[:, 0].clone()
        prev_losses[prev_losses < 1e-8] = 1e-8  # prevent div by 0

        # Raw improvement ratio
        ratios = current_losses / prev_losses  # L(t-1) / L(t-2)

        # Normalize with current loss magnitude (to downscale tiny-loss tasks)
        normalized_ratios = ratios * (current_losses / current_losses.mean())

        # Clamp for numerical stability
        scaled = torch.clamp(normalized_ratios / self.temperature, max=50)

        # Softmax-style weight computation
        exp_scaled = torch.exp(scaled)
        weights = exp_scaled / exp_scaled.sum() * len(current_losses)

        # Sanity check
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("[Warning] NaN or Inf in task weights — reverting to uniform weights.")
            weights = torch.ones_like(weights)

        self.task_weights = weights.detach()

        # Update history
        self.loss_history[:, 0] = self.loss_history[:, 1]
        self.loss_history[:, 1] = current_losses.detach()

    def forward(self, losses, epoch):
        losses_tensor = torch.stack(losses)

        if epoch >= 2:
            with torch.no_grad():
                self.update_weights(losses_tensor.detach())
        else:
            self.task_weights = torch.ones(len(losses), device=self.loss_history.device)

        weighted_loss = self.task_weights * losses_tensor

        if torch.isnan(weighted_loss).any():
            print("[Error] NaN in weighted loss")
            print("Losses:", losses_tensor)
            print("Weights:", self.task_weights)

        return weighted_loss.sum()
        
class ManualWeightedLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.task_weights = torch.tensor(weights)  # Provide as a list or tensor

    def forward(self, losses):
        weighted_losses = self.task_weights.to(losses[0].device) * torch.stack(losses)
        return weighted_losses.sum()