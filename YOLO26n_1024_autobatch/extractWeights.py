from ultralytics import YOLO


def train_deepthyro():
    # Load YOLO26 Nano model (or Large as per your research notes)
    model = YOLO("yolo26n.pt")

    # Start Training
    results = model.train(
        data="data.yaml",
        epochs=500,
        imgsz=1024,
        batch=16,
        optimizer="MuSGD",
        device=0,
        project="DeepThyro",
        name="YOLO26_1024_baseline",
        patience=25,
        plots=True,
        save=True,
        val=True,
        # results=model.train(resume=True),  # in case of last.py
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


def test_deepthyro():

    model = YOLO("DeepThyro7.pt")
    # model = YOLO("Deepthyro6.pt")

    results = model.val(
        data="data.yaml",
        split="test",
        imgsz=1024,
        device=0,
        plots=True,
        batch=16,
    )

    # Extract the metrics specifically for the test run
    metrics = results.results_dict
    p = metrics.get("metrics/precision(B)", 0)
    r = metrics.get("metrics/recall(B)", 0)
    m50 = metrics.get("metrics/mAP50(B)", 0)
    m95 = metrics.get("metrics/mAP50-95(B)", 0)

    # Calculate F1 Score
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    print("\n" + "=" * 45)
    print("DEEPTHYRO FINAL TEST SET EVALUATION:")
    print("=" * 45)
    print(f"Precision:    {p:.4f}")
    print(f"Recall:       {r:.4f}")
    print(f"mAP@50:       {m50:.4f}")
    print(f"mAP@50-95:    {m95:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("=" * 45)
    print("Check your 'runs/detect/test' ")


if __name__ == "__main__":
    # train_deepthyro()
    test_deepthyro()
