import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO

model = YOLO("best.pt")


plt.style.use("dark_background")
sns.set_context("paper", font_scale=1.2)


def generate_annotations(image_file):
    """
    Runs YOLO inference on a single image and saves the structural
    JSON logging to a matching .json file.
    """
    json_outputfile = image_file.replace(".jpg", ".json")

    results = model.predict(
        source=image_file,
        imgsz=1024,
        device=0,
        conf=0.3,
        verbose=False,
    )[0]

    json_predictions = results.to_json()
    parsed_json = json.loads(json_predictions)

    with open(json_outputfile, "w", encoding="utf-8") as json_file:
        json.dump(parsed_json, json_file, indent=4)

    print(f"[Success] Structural prediction logging complete: {json_outputfile}")


def draw_predictions(image_path, json_path):
    """
    Reads the image and JSON, draws bounding boxes with premium alpha-blended overlays.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not os.path.exists(json_path):
        return img

    with open(json_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    for pred in predictions:
        box = pred["box"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        cls_name = pred["name"]
        conf = pred["confidence"]

        if cls_name == "001970":  # Malignant
            label = f"Malignant {conf:.2f}"
            color = (255, 107, 107)
        else:  # Benign (001131)
            label = f"Benign {conf:.2f}"
            color = (0, 168, 168)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        overlay = img.copy()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            overlay, (x1, y1 - text_h - 12), (x1 + text_w + 10, y1), color, -1
        )

        # Apply the transparency (0.6 opacity)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.putText(
            img,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return img


def create_board(directory, output_filename, title):
    """
    Compiles a sleek, zero-margin 6-scan grid.
    """
    image_files = glob.glob(os.path.join(directory, "*.jpg"))

    if not image_files:
        print(f"[Warning] No images found in {directory}")
        return

    image_files = image_files[:6]
    n_images = len(image_files)

    cols = 3
    rows = int(np.ceil(n_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), facecolor="#121212")

    fig.suptitle(title, fontsize=22, weight="bold", color="#E0E0E0", y=0.96)

    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img_path in enumerate(image_files):
        json_path = img_path.replace(".jpg", ".json")

        if not os.path.exists(json_path):
            generate_annotations(img_path)

        annotated_img = draw_predictions(img_path, json_path)

        axes[i].imshow(annotated_img)
        axes[i].axis("off")

        axes[i].set_title(
            os.path.basename(img_path), fontsize=11, color="#A0A0A0", pad=8
        )

    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        wspace=0.05, hspace=0.1, top=0.88, bottom=0.05, left=0.02, right=0.98
    )

    plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="#121212")
    plt.close()
    print(f"[Success] Validation board saved -> {output_filename}")


def sucess_boards(success_dir):
    print(f"--- Generating Success Board ---")
    parent_dir = os.path.dirname(success_dir)
    output_path = os.path.join(parent_dir, "success_board.jpg")
    create_board(success_dir, output_path, "DEEPTHYRO: SUCCESS Cases")


def fail_boards(fail_dir):
    print(f"--- Generating Failure Board ---")
    parent_dir = os.path.dirname(fail_dir)
    output_path = os.path.join(parent_dir, "failure_board.jpg")
    create_board(fail_dir, output_path, "DEEPTHYRO: FAILURE cases and errors")


if __name__ == "__main__":
    success_dir = "thyroid_board/success"
    fail_dir = "thyroid_board/failure"

    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    sucess_boards(success_dir)
    fail_boards(fail_dir)
