from ultralytics import YOLO
import torch


# model = YOLO("yolo26x.pt")
model = YOLO("models/best.pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

results = model.predict('input_videos/football_game1.mp4',
                        save=True, device=device)

print(results[0])
print("================")
for box in results[0].boxes:
    print(box)
