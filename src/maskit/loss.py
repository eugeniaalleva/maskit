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

class DynamicWeightAveragingLoss(nn.Module):
    def __init__(self, num_tasks, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("loss_history", torch.ones(num_tasks, 2))  # L_{t-2}, L_{t-1}
        self.register_buffer("task_weights", torch.ones(num_tasks))

    def update_weights(self, current_losses):
        ratios = current_losses / (self.loss_history[:, 0] + 1e-8)
        exp_ratios = torch.exp(ratios / self.temperature)
        self.task_weights = (len(current_losses) * exp_ratios / exp_ratios.sum()).detach()

        # Update history
        self.loss_history[:, 0] = self.loss_history[:, 1]
        self.loss_history[:, 1] = current_losses.detach()

    def forward(self, losses, epoch):
        if epoch >= 2:
            current_losses = torch.stack(losses).detach()
            self.update_weights(current_losses)
        else:
            self.task_weights = torch.ones(len(losses), device=self.loss_history.device)  # Equal weighting for first 2 epochs

        total_loss = torch.sum(self.task_weights * torch.stack(losses))
        return total_loss

class ManualWeightedLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.task_weights = torch.tensor(weights)  # Provide as a list or tensor

    def forward(self, losses):
        weighted_losses = self.task_weights.to(losses[0].device) * torch.stack(losses)
        return weighted_losses.sum()