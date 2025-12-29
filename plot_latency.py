import glob
import os
import numpy as np
import matplotlib.pyplot as plt

DISPLAY_NAME = {
    "latency_yolo": "YOLOv8n",
    "latency_ssd_v1_quant": "SSD MobileNet V1 (8-bit)",
    "latency_ssd_v2_quant": "SSD MobileNet V2 (8-bit)",
    "latency_ssd_v2_fpnlite_int8": "SSD MobileNet V2 FPN-Lite (8-bit)",
}

FILES = sorted(glob.glob("latency_*.npy"))

SSD_ONLY_KEYS = [
    "latency_ssd_v1_quant",   
    "latency_ssd_v2_quant",    
]

def stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    m = float(arr.mean())
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    fps = float(1000.0 / m) if m > 0 else float("inf")
    return m, p50, p95, fps

def load_series(file_list):
    labels, series = [], []
    for f in file_list:
        arr = np.load(f)
        base = os.path.splitext(os.path.basename(f))[0]  # e.g. latency_yolo
        label = DISPLAY_NAME.get(base, base.replace("latency_", ""))
        labels.append(label)
        series.append(arr)
    return labels, series

def main():
    if not FILES:
        print("No latency_*.npy found. Please run bench scripts first.")
        return

    # -------- 讀取全模型 --------
    labels_all, series_all = load_series(FILES)

    print("\n=== Latency Summary (ms) ===")
    for label, arr in zip(labels_all, series_all):
        m, p50, p95, fps = stats(arr)
        print(f"- {label}")
        print(f"  runs={len(arr)}  mean={m:.3f}  p50={p50:.3f}  p95={p95:.3f}  fps_est={fps:.2f}")

    # ======================================================
    # 圖 1：Histogram（全模型延遲分佈）
    # ======================================================
    plt.figure(figsize=(9, 5))
    for label, arr in zip(labels_all, series_all):
        plt.hist(arr, bins=30, alpha=0.55, label=label)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Inference Latency Distribution (All Models)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    # 圖 2：Boxplot（全模型）
    # ======================================================
    plt.figure(figsize=(9, 5))
    plt.boxplot(series_all, labels=labels_all, showfliers=True)
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Boxplot (All Models)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    # ======================================================
    # 圖 3：Boxplot（只比較 SSD V1 / SSD V2）
    # ======================================================
    ssd_files = []
    for key in SSD_ONLY_KEYS:
        fname = key + ".npy"
        if os.path.exists(fname):
            ssd_files.append(fname)

    if len(ssd_files) < 2:
        print("\n[WARN] SSD-only boxplot skipped: cannot find BOTH files:")
        for key in SSD_ONLY_KEYS:
            print(" -", key + ".npy")
        print("請確認你的檔名是否為上述兩個，或修改 SSD_ONLY_KEYS。")
        return

    labels_ssd, series_ssd = load_series(ssd_files)

    plt.figure(figsize=(6, 5))
    plt.boxplot(series_ssd, labels=labels_ssd, showfliers=True)
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Boxplot (SSD V1 vs SSD V2)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
