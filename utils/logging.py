import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.step = 0

    def log_scalar(self, name: str, value: float, step: int = None):
        self.writer.add_scalar(name, value, step if step is not None else self.step)

    def log_hyperparams(self, hparams: dict):
        text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        self.writer.add_text("hyperparameters", text, global_step=0)

    def increment_step(self):
        self.step += 1

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


# Utility function for creating log directory

def create_log_dir(base_path: str = "logs") -> Path:
    log_dir = Path(base_path) / f"run_{len(os.listdir(base_path))}"
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir
