import os
from pathlib import Path

MODEL_PATH = Path(os.getcwd()) / "bookwise_model"
MODEL_PATH.mkdir(parents=True, exist_ok=True)