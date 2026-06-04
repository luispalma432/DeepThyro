from ultralytics import YOLO

# testImagePath = "0413.jpg" #639*534
# testImagePath = "0330.jpg"  # 879*744
testImagePath = "0330.jpg"  # 1237*730


model = "DeepThyro7.pt"


def benchmark_realtime_latency(image_path, model_path):

    model = YOLO(model_path)

    for _ in range(3):
        _ = model.predict(image_path, imgsz=1024, verbose=False, device=0)

    results = model.predict(image_path, imgsz=1024, verbose=False, device=0)

    speeds = results[0].speed

    prep_ms = speeds.get("preprocess", 0)
    inf_ms = speeds.get("inference", 0)
    post_ms = speeds.get("postprocess", 0)

    total_ms = prep_ms + inf_ms + post_ms

    # Calculate Frames Per Second (FPS)
    fps = 1000 / total_ms if total_ms > 0 else 0

    print("\n" + "=" * 45)
    print(f"{'DEEPTHYRO LATENCY & FPS REPORT':^45}")
    print("=" * 45)
    print(f"Resolution:      1024x1024 uncompressed")
    print("-" * 45)
    print(f"Preprocessing:   {prep_ms:>6.2f} ms")
    print(f"Inference:       {inf_ms:>6.2f} ms")
    print(f"Postprocessing:  {post_ms:>6.2f} ms")
    print("-" * 45)
    print(f"Total Latency:   {total_ms:>6.2f} ms per frame")
    print(f"Estimated FPS:   {fps:>6.1f} Frames Per Second")
    print("=" * 45)

    # Clinical Viability Check
    if fps >= 30:
        print("\nVerdict: FULL REAL-TIME VIABLE.")
        print("Suitable for smooth clinical video streams (>30 FPS).")
    elif fps >= 15:
        print("\nVerdict: MARGINALLY REAL-TIME.")
        print(
            "Usable for clinical tracking, but feed may feel slightly sluggish (15-30 FPS)."
        )
    else:
        print("\nVerdict: NOT REAL-TIME.")
        print(
            "Will cause frame-stutter in live video. Consider TensorRT or OpenVINO optimization."
        )


if __name__ == "__main__":
    # Point this to any random ultrasound image on your drive

    benchmark_realtime_latency(testImagePath, model)
