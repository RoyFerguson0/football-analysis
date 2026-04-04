from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import dotenv_values, load_dotenv
import shutil
import os
import torch
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def main():
    print(get_project_root() / ".env")
    secrets = dotenv_values(get_project_root() / ".env")

    rf = Roboflow(api_key=secrets["ROBOFLOW_API_KEY"])
    project = rf.workspace(
        "roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)
    dataset = version.download("yolo26")

    import gc

    # Optional cleanup to free up memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo26x.pt")

    model.train(data=f"{dataset.location}/data.yaml",
                epochs=100, imgsz=640, batch=8, workers=4, device=device)


if __name__ == "__main__":
    main()
