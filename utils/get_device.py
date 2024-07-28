import torch


def get_device(device: str = "cpu") -> torch.device:
    DEVICE = torch.device("cpu")

    if device.lower() == "mps" and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif device.lower() == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    return DEVICE
