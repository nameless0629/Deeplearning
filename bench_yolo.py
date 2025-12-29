import time
import numpy as np
import cv2
from ultralytics import YOLO
import glob

MODEL_PATH = "yolov8n.pt"
IMG_DIR = "test_images"
RUNS = 100

model = YOLO(MODEL_PATH)

imgs = glob.glob(f"{IMG_DIR}/*.jpg")
imgs = imgs[:RUNS]

times = []

# warm-up
for _ in range(10):
    _ = model(imgs[0], device="cpu", verbose=False)

for img_path in imgs:
    t0 = time.perf_counter()
    _ = model(img_path, device="cpu", verbose=False)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)  # ms

times = np.array(times)
np.save("latency_yolo.npy", times)

print("YOLO done")
print("mean_ms:", times.mean())
print("p50_ms:", np.percentile(times, 50))
print("p95_ms:", np.percentile(times, 95))
print("fps_est:", 1000 / times.mean())
