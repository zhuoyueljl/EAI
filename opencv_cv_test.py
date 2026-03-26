from __future__ import annotations

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs"
INPUT_IMAGE = ASSETS_DIR / "input_demo.png"
OUTPUT_IMAGE = OUTPUTS_DIR / "opencv_processed.png"
VIS_IMAGE = OUTPUTS_DIR / "opencv_visualization.png"


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def create_demo_image(path: Path, width: int = 960, height: int = 640) -> None:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    blue = (255 * (1.0 - grid_x)).astype(np.uint8)
    green = (255 * grid_y).astype(np.uint8)
    red = (255 * (0.35 + 0.65 * np.sin(np.pi * grid_x) * np.cos(np.pi * grid_y))).clip(0, 255).astype(np.uint8)
    image = np.dstack([blue, green, red])

    cv2.putText(image, "OpenCV cv_test", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.rectangle(image, (70, 140), (360, 380), (40, 40, 40), -1)
    cv2.circle(image, (660, 220), 95, (0, 255, 255), -1)
    cv2.line(image, (420, 500), (880, 120), (255, 255, 255), 7)
    cv2.putText(image, "Synthetic input", (80, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(str(path), image)


def benchmark_and_process(image_bgr: np.ndarray, loops: int = 100) -> tuple[np.ndarray, dict[str, float]]:
    timings_ms = {"grayscale": 0.0, "gaussian_blur": 0.0, "canny_edge": 0.0}
    gray = None
    blur = None
    edges = None

    for _ in range(loops):
        start = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        timings_ms["grayscale"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        timings_ms["gaussian_blur"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        edges = cv2.Canny(blur, 80, 160)
        timings_ms["canny_edge"] += (time.perf_counter() - start) * 1000

    assert gray is not None and blur is not None and edges is not None

    average_ms = {name: value / loops for name, value in timings_ms.items()}
    total_ms = sum(average_ms.values())
    fps = 1000.0 / total_ms if total_ms > 0 else float("inf")

    colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.78, colored_edges, 0.22, 0.0)

    text_rows = [
        f"Read shape: {image_bgr.shape[1]}x{image_bgr.shape[0]}",
        f"Gray avg: {average_ms['grayscale']:.3f} ms",
        f"Blur avg: {average_ms['gaussian_blur']:.3f} ms",
        f"Canny avg: {average_ms['canny_edge']:.3f} ms",
        f"Pipeline FPS: {fps:.2f}",
    ]
    for idx, row in enumerate(text_rows):
        cv2.putText(
            overlay,
            row,
            (20, 35 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

    metrics = dict(average_ms)
    metrics["fps"] = fps
    return overlay, metrics


def save_visualization(original_bgr: np.ndarray, processed_bgr: np.ndarray, metrics: dict[str, float]) -> None:
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(processed_rgb)
    axes[1].set_title("Processed + Timing")
    axes[1].axis("off")

    fig.suptitle(
        "OpenCV Read / Process / Save Demo\n"
        f"gray={metrics['grayscale']:.3f} ms | blur={metrics['gaussian_blur']:.3f} ms | "
        f"canny={metrics['canny_edge']:.3f} ms | fps={metrics['fps']:.2f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(VIS_IMAGE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def show_interactive_results(original_bgr: np.ndarray, processed_bgr: np.ndarray, metrics: dict[str, float]) -> None:
    views = [
        ("Original", cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)),
        ("Processed + Timing", cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)),
    ]
    instructions = "Left/Right switch | s save current view | q close"
    state = {"index": 0}

    fig, ax = plt.subplots(figsize=(12, 7))

    def render() -> None:
        title, image = views[state["index"]]
        ax.clear()
        ax.imshow(image)
        ax.set_title(
            f"{title}\n"
            f"gray={metrics['grayscale']:.3f} ms | blur={metrics['gaussian_blur']:.3f} ms | "
            f"canny={metrics['canny_edge']:.3f} ms | fps={metrics['fps']:.2f}\n"
            f"{instructions}"
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("right", "d"):
            state["index"] = (state["index"] + 1) % len(views)
            render()
        elif event.key in ("left", "a"):
            state["index"] = (state["index"] - 1) % len(views)
            render()
        elif event.key == "s":
            export_path = OUTPUTS_DIR / f"interactive_view_{state['index']}.png"
            fig.savefig(export_path, dpi=160, bbox_inches="tight")
            print(f"Saved interactive view: {export_path}")
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


def main() -> None:
    ensure_dirs()

    if not INPUT_IMAGE.exists():
        create_demo_image(INPUT_IMAGE)

    image_bgr = cv2.imread(str(INPUT_IMAGE))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {INPUT_IMAGE}")

    processed_bgr, metrics = benchmark_and_process(image_bgr, loops=100)
    cv2.imwrite(str(OUTPUT_IMAGE), processed_bgr)
    save_visualization(image_bgr, processed_bgr, metrics)

    print("OpenCV demo finished.")
    print(f"Input image:  {INPUT_IMAGE}")
    print(f"Output image: {OUTPUT_IMAGE}")
    print(f"Visual image: {VIS_IMAGE}")
    print("Average timings (ms):")
    print(f"  grayscale:     {metrics['grayscale']:.3f}")
    print(f"  gaussian_blur: {metrics['gaussian_blur']:.3f}")
    print(f"  canny_edge:    {metrics['canny_edge']:.3f}")
    print(f"Pipeline FPS:    {metrics['fps']:.2f}")
    print("Interactive window: use Left/Right to switch, s to save, q to quit.")

    show_interactive_results(image_bgr, processed_bgr, metrics)


if __name__ == "__main__":
    main()
