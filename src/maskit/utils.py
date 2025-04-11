import torch

def move_to_device(value, device):
    'Move nested dictionary entris to device.'
    if isinstance(value, dict):
        return {k: v.to(device) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):
        return value.to(device)
    else:
        raise RuntimeError(f"no processing possible")


def build_task_label_mappings(task_to_labels):
    """
    Given a dictionary of task -> list of label words (verbalizers),
    returns consistent mappings for task and label IDs.
    """
    # Sort tasks and labels to fix the order
    sorted_tasks = sorted(task_to_labels.keys())

    task_to_id = {task: idx for idx, task in enumerate(sorted_tasks)}
    id_to_task = {idx: task for task, idx in task_to_id.items()}

    label_to_id = {}
    id_to_label = {}
    for task in sorted_tasks:
        labels = sorted(task_to_labels[task])  # ensure consistent label ordering
        label_to_id[task] = {label: i for i, label in enumerate(labels)}
        id_to_label[task] = {i: label for label, i in label_to_id[task].items()}

    return task_to_id, id_to_task, label_to_id, id_to_label