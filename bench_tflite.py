import os, glob, time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

IMG_DIR = "test_images"
WARMUP = 10
RUNS = 100
NUM_THREADS = 4

MODELS = [
    ("ssd_v1_quant", "ssd_mobilenet_v1_quant.tflite"),
    #("ssd_v2_fpnlite_int8", "ssd_mobilenet_v2_fpnlite_035_224_int8.tflite"),
    ("ssd_v2_quant", "mobilenet_ssd_v2_coco_quant_postprocess.tflite"),
]

def list_images(img_dir: str, runs: int):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(glob.glob(os.path.join(img_dir, ext)))
    imgs = sorted(imgs)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return imgs[: min(runs, len(imgs))]

def preprocess_for_tflite(img_bgr, w, h, dtype, quant):
    # resize
    img = cv2.resize(img_bgr, (w, h))
    img = img.astype(np.float32)

    scale, zero = quant
    # 某些 float 模型 quant=(0.0, 0)；我們只處理常見 uint8/int8/float32
    if dtype == np.float32:
        # 常見是 0~1
        img = img / 255.0
        return img.astype(np.float32)

    # 量化模型（uint8 / int8）
    img = img / 255.0
    if scale == 0:
        # 保底避免除 0（少見）
        scale = 1e-6

    q = np.round(img / scale + zero)

    if dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    return q

def bench_one(model_path: str, imgs, warmup=WARMUP, runs=RUNS, num_threads=NUM_THREADS):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    idx = inp["index"]
    dtype = inp["dtype"]
    quant = inp.get("quantization", (0.0, 0))

    # shape: [1, H, W, 3]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])

    # warm-up 用第一張
    img0 = cv2.imread(imgs[0])
    x0 = preprocess_for_tflite(img0, w, h, dtype, quant)
    interpreter.set_tensor(idx, x0[None, ...])
    for _ in range(warmup):
        interpreter.invoke()

    times = []
    for i, p in enumerate(imgs[:runs]):
        im = cv2.imread(p)
        x = preprocess_for_tflite(im, w, h, dtype, quant)
        interpreter.set_tensor(idx, x[None, ...])

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return {
        "input_name": inp.get("name", ""),
        "input_dtype": str(dtype),
        "input_hw": f"{h}x{w}",
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps_est": float(1000.0 / times.mean()),
        "latencies": times,
    }

def main():
    imgs = list_images(IMG_DIR, RUNS)
    rows = []

    for tag, path in MODELS:
        if not os.path.exists(path):
            print(f"[SKIP] {tag}: file not found -> {path}")
            continue

        print(f"\n=== Running: {tag} ({path}) ===")
        r = bench_one(path, imgs)
        npy_name = f"latency_{tag}.npy"
        np.save(npy_name, r["latencies"])

        row = {
            "model_tag": tag,
            "model_path": path,
            "input_name": r["input_name"],
            "input_dtype": r["input_dtype"],
            "input_hw": r["input_hw"],
            "mean_ms": r["mean_ms"],
            "p50_ms": r["p50_ms"],
            "p95_ms": r["p95_ms"],
            "fps_est": r["fps_est"],
            "runs": len(r["latencies"]),
        }
        rows.append(row)

        print("mean_ms:", r["mean_ms"])
        print("p50_ms :", r["p50_ms"])
        print("p95_ms :", r["p95_ms"])
        print("fps_est:", r["fps_est"])
        print("saved  :", npy_name)

    if len(rows) == 0:
        print("\nNo models were run. Check filenames in MODELS.")
        return

    df = pd.DataFrame(rows).sort_values("mean_ms")
    df.to_csv("tflite_bench_summary.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: tflite_bench_summary.csv")
    print(df)

if __name__ == "__main__":
    main()
