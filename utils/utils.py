from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np


ImageMode = Literal["gray", "rgb"]


def to_float32(img: np.ndarray) -> np.ndarray:
    # Convert image to float32
    if img.dtype == np.float32:
        return np.ascontiguousarray(img)

    # Normalize integer images to [0, 1]
    if np.issubdtype(img.dtype, np.integer):
        return np.ascontiguousarray(img.astype(np.float32) / 255.0)

    # Convert other types to float32
    return np.ascontiguousarray(img.astype(np.float32))


def to_uint8(img: np.ndarray) -> np.ndarray:
    # Return as is if already uint8
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)

    out = img.astype(np.float32)

    # Scale normalized images to [0, 255]
    if out.min() >= 0.0 and out.max() <= 1.0:
        out = out * 255.0

    # Clip values to valid uint8 range
    out = np.clip(out, 0.0, 255.0)

    return np.ascontiguousarray(out.astype(np.uint8))


def load_image(
    path: str | Path,
    mode: ImageMode = "rgb",
    as_float: bool = False,
) -> np.ndarray:
    # Convert path to Path object
    path = Path(path)

    # Load grayscale image
    if mode == "gray":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        # Load color image and convert BGR to RGB
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make array contiguous
    img = np.ascontiguousarray(img)

    # Convert to float32 if requested
    if as_float:
        img = to_float32(img)

    return img


def save_image(
    path: str | Path,
    img: np.ndarray,
    input_mode: ImageMode = "rgb",
) -> None:
    # Convert path to Path object
    path = Path(path)

    # Force PNG extension
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    # Convert image to uint8
    out = to_uint8(img)

    # Save grayscale image
    if out.ndim == 2:
        cv2.imwrite(str(path), out)

    # Save RGB image after converting to BGR
    elif out.ndim == 3 and out.shape[2] == 3:
        if input_mode == "rgb":
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), out)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    # Return a copy if image is already grayscale
    if img.ndim == 2:
        return np.ascontiguousarray(img.copy())

    # Convert RGB image to grayscale
    img_f = img.astype(np.float32)
    gray = (
        0.2989 * img_f[..., 0]
        + 0.5870 * img_f[..., 1]
        + 0.1140 * img_f[..., 2]
    )

    # Return uint8 if input is uint8
    if img.dtype == np.uint8:
        return np.ascontiguousarray(np.clip(gray, 0, 255).astype(np.uint8))

    # Return float32 otherwise
    return np.ascontiguousarray(gray.astype(np.float32))


def show_image(
    img: np.ndarray,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
    axis: bool = False,
    cmap: str | None = None,
) -> None:
    # Create figure
    plt.figure(figsize=figsize)

    # Show grayscale image
    if img.ndim == 2:
        if np.issubdtype(img.dtype, np.floating):
            plt.imshow(img, cmap=cmap or "gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img, cmap=cmap or "gray", vmin=0, vmax=255)

    # Show RGB image
    elif img.ndim == 3 and img.shape[2] == 3:
        if img.dtype == np.uint8:
            plt.imshow(img)
        else:
            plt.imshow(np.clip(img, 0.0, 1.0))

    # Set title if provided
    if title is not None:
        plt.title(title)

    # Hide axis by default
    if not axis:
        plt.axis("off")

    # Adjust layout and render image
    plt.tight_layout()
    plt.show()

