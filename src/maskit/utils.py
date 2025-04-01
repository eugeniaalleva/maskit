import torch

def move_to_device(value, device):
    'Move nested dictionary entris to device.'
    if isinstance(value, dict):
        return {k: v.to(device) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):
        return value.to(device)
    else:
        raise RuntimeError(f"no processing possible")
