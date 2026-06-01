from ultralytics import YOLO


def train_deepthyro():
    # Load YOLO26 Nano model (or Large as per your research notes)
    model = YOLO("last3.pt")

    # Start Training
    results = model.train(
        data="data.yaml",
        epochs=200,
        imgsz=1024,
        batch=-1,
        optimizer="MuSGD",
        device=0,
        project="DeepThyro",
        name="YOLO26_1024_baseline",
        patience=25,
        plots=True,
        save=True,
        val=True,
        results=model.train(resume=True),  # in case of last.py
        # Disable online augmentations (since data is already pre-augmented)
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,
    )

    # metrics are stored in the results.results_dict
    metrics = results.results_dict
    p = metrics["metrics/precision(B)"]
    r = metrics["metrics/recall(B)"]
    m50 = metrics["metrics/mAP50(B)"]
    m95 = metrics["metrics/mAP50-95(B)"]

    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    print("\n" + "=" * 40)
    print("DEEPTHYRO FINAL EVALUATION:")
    print(f"Precision:    {p:.4f}")
    print(f"Recall:       {r:.4f}")

    print(f"mAP@50:       {m50:.4f}")
    print(f"mAP@50-95:    {m95:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("=" * 40)
    print(f"Graphs saved in: DeepThyro/YOLO26_1024_PreAugmented/results.png")


if __name__ == "__main__":
    train_deepthyro()


"""
YOLO26 range
box_loss: 0.9725
cls_loss:0.5773
dfl_loss:0.01045
"""


"""
yolov11 range
box_loss:1.049
cls_loss: 0.8169
dfl_loss:1.237
"""
