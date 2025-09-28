"""
visualize.py
------------
Create ROC curve, confusion matrix heatmap, and TP/FP/FN/TN image grids
for the lens demo.

Inputs:
- --data-root: folder with test.csv and images (default: data_scientific_test)
- --predictions: CSV with columns either:
    (A) y_true, y_prob, y_pred              [from eval.py -> results/test_probs.csv]
    (B) filepath, y_true, y_prob, y_pred    [from eval.py -> results/detailed_predictions.csv]

Outputs are saved under ./results:
- roc_curve.png
- confusion_matrix.png
- tp_grid.png, fp_grid.png, fn_grid.png, tn_grid.png

Usage (PowerShell):
    .\venv\Scripts\Activate
    pip install matplotlib scikit-learn pandas pillow numpy
    py src\visualize.py --data-root data_scientific_test --predictions results\test_probs.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


def _load_predictions(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    cols = {c.lower(): c for c in df.columns}

    # Normalize expected column names
    # y_true / y_prob / y_pred are required; filepath is optional
    def pick(name: str) -> str:
        for k, v in cols.items():
            if k == name:
                return v
        raise ValueError(f"Column '{name}' not found in {pred_csv}. Found: {list(df.columns)}")

    y_true_col = pick("y_true")
    y_prob_col = pick("y_prob")
    # y_pred sometimes not saved; we can compute at 0.5 if missing
    y_pred_col = cols.get("y_pred")

    out = pd.DataFrame({
        "y_true": df[y_true_col].astype(int),
        "y_prob": df[y_prob_col].astype(float),
    })
    if y_pred_col:
        out["y_pred"] = df[y_pred_col].astype(int)
    else:
        out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)

    # Keep filepath if present
    fp_col = cols.get("filepath")
    if fp_col:
        out["filepath"] = df[fp_col].astype(str)

    return out


def _attach_filepaths(preds: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """
    Ensure we have filepaths column.
    If predictions CSV didn't include filepaths, join by row order with test.csv.
    """
    if "filepath" in preds.columns:
        return preds

    test_csv = data_root / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at {test_csv}")

    test_df = pd.read_csv(test_csv)
    if "filepath" not in test_df.columns:
        raise ValueError(f"'filepath' column missing in {test_csv}")

    if len(test_df) != len(preds):
        # Fall back to inner join on y_true positions (not ideal)
        # but usually eval kept the original test order → lengths match.
        raise ValueError("Row count mismatch between predictions and test.csv; "
                         "rerun eval so ordering matches test.csv.")
    preds = preds.copy()
    preds["filepath"] = test_df["filepath"].astype(str)
    return preds


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 5), dpi=140)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return auc


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-lens (0)", "Lens (1)"])
    plt.figure(figsize=(4.5, 4.5), dpi=140)
    disp.plot(values_format="d", cmap=None, colorbar=False)  # use default colors
    plt.title("Confusion Matrix @ 0.5")
    plt.tight_layout()
    # ConfusionMatrixDisplay creates its own figure; grab current figure to save
    plt.savefig(out_path)
    plt.close()


def _read_image(img_path: Path, size: int = 64) -> np.ndarray:
    # Robust to grayscale/RGB; always return HxWx3 in [0,1]
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((size, size))
        arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def _make_grid(images: list[np.ndarray], cols: int = 4, pad: int = 2) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), dtype=np.float32)

    h, w, c = images[0].shape
    rows = int(np.ceil(len(images) / cols))
    grid = np.ones((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, c), dtype=np.float32)

    grid[:] = 1.0  # white background
    idx = 0
    for r in range(rows):
        for cl in range(cols):
            if idx >= len(images):
                break
            y0 = r * (h + pad)
            x0 = cl * (w + pad)
            grid[y0:y0 + h, x0:x0 + w] = images[idx]
            idx += 1
    return grid


def save_example_grid(df: pd.DataFrame, mask: np.ndarray, data_root: Path,
                      out_path: Path, title: str, max_n: int = 16, size: int = 96) -> int:
    idxs = np.where(mask)[0][:max_n].tolist()
    if not idxs:
        return 0

    ims = []
    for i in idxs:
        p = data_root / df.iloc[i]["filepath"]
        try:
            ims.append(_read_image(p, size=size))
        except Exception:
            # Skip unreadable images
            continue

    if not ims:
        return 0

    grid = _make_grid(ims, cols=4, pad=4)
    plt.figure(figsize=(6, 6), dpi=140)
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return len(ims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data_scientific_test")
    ap.add_argument("--predictions", type=str, required=True,
                    help="Path to predictions CSV (test_probs.csv or detailed_predictions.csv)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pred_csv = Path(args.predictions)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv}")

    # Load predictions and ensure filepaths available
    preds = _load_predictions(pred_csv)
    preds = _attach_filepaths(preds, data_root)

    y_true = preds["y_true"].to_numpy().astype(int)
    y_prob = preds["y_prob"].to_numpy().astype(float)
    y_pred = preds["y_pred"].to_numpy().astype(int)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) ROC
    try:
        auc = plot_roc(y_true, y_prob, results_dir / "roc_curve.png")
        print(f"ROC AUC: {auc:.3f}  →  results/roc_curve.png")
    except Exception as e:
        print("ROC failed (maybe only one class present). Details:", e)

    # 2) Confusion matrix
    plot_confusion(y_true, y_pred, results_dir / "confusion_matrix.png")
    print("Saved confusion matrix → results/confusion_matrix.png")

    # 3) Example grids
    masks = {
        "tp": (y_true == 1) & (y_pred == 1),
        "fp": (y_true == 0) & (y_pred == 1),
        "fn": (y_true == 1) & (y_pred == 0),
        "tn": (y_true == 0) & (y_pred == 0),
    }
    for name, m in masks.items():
        saved = save_example_grid(preds, m, data_root, results_dir / f"{name}_grid.png",
                                  title=name.upper(), max_n=16, size=96)
        if saved > 0:
            print(f"Saved {name.upper()} grid ({saved} images) → results/{name}_grid.png")
        else:
            print(f"No samples to plot for {name.upper()} grid.")

    print("\nAll visualizations saved under ./results")


if __name__ == "__main__":
    main()
