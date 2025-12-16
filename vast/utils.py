import logging
import yaml
from pathlib import Path
import torch

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("VAST")


def load_yaml(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"The YAML file {path} is empty or invalid.")
    return data



def get_device():
    if torch.cuda.is_available():
        print("Using NVIDIA GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("No GPU found → using CPU")
        return torch.device("cpu")

