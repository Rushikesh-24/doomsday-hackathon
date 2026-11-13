"""Image decoding, validation and preprocessing."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}  # MPO = multi-picture JPEG from phones
Image.MAX_IMAGE_PIXELS = 12000 * 12000  # decompression-bomb guard


class InvalidImageError(ValueError):
    pass


@dataclass
class PreparedImage:
    name: str
    rgb: np.ndarray            # full-resolution, EXIF-corrected RGB (H, W, 3) uint8
    infer_rgb: np.ndarray      # possibly downscaled copy used for inference
    scale: float               # infer -> original coordinate factor (>=1)
    format: str | None
    exif_transposed: bool

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]


def load_image(data: bytes | str | Path, name: str | None = None, *, max_bytes: int | None = None,
               min_side: int = 32, max_side: int = 12000, max_infer_side: int = 1920) -> PreparedImage:
    """Decode + validate + preprocess.

    * rejects non-images, unsupported formats, truncated files, too-small/too-large images
    * applies EXIF orientation (phone photos) so boxes match what a viewer displays
    * converts any mode (L, RGBA, P, CMYK, 16-bit) to 8-bit RGB
    * downsizes very large photos for inference; boxes are later mapped back by `scale`
    """
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.is_file():
            raise InvalidImageError(f"file not found: {p}")
        name = name or p.name
        data = p.read_bytes()
    name = name or "image"
    if not data:
        raise InvalidImageError(f"{name}: empty file")
    if max_bytes and len(data) > max_bytes:
        raise InvalidImageError(f"{name}: file too large ({len(data) / 1e6:.1f} MB > {max_bytes / 1e6:.1f} MB)")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()  # forces full decode -> catches truncated files
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"{name}: not a decodable image ({e.__class__.__name__}: {e})") from e
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"{name}: unsupported format {fmt}; allowed: {sorted(ALLOWED_FORMATS)}")

    before = img.size
    orientation = img.getexif().get(0x0112, 1) if hasattr(img, "getexif") else 1
    img = ImageOps.exif_transpose(img)
    transposed = orientation not in (1, None)

    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < min_side:
        raise InvalidImageError(f"{name}: image too small ({w}x{h}); minimum side is {min_side}px")
    if max(w, h) > max_side:
        raise InvalidImageError(f"{name}: image too large ({w}x{h}); maximum side is {max_side}px")

    rgb = np.asarray(img, dtype=np.uint8)
    scale = 1.0
    infer = rgb
    if max(w, h) > max_infer_side:
        scale = max(w, h) / max_infer_side
        small = img.resize((round(w / scale), round(h / scale)), Image.BILINEAR)
        infer = np.asarray(small, dtype=np.uint8)
        scale = w / infer.shape[1]
    log.debug("%s: %s %s -> %dx%d (infer scale %.2f)", name, fmt, before, w, h, scale)
    return PreparedImage(name=name, rgb=rgb, infer_rgb=infer, scale=scale, format=fmt, exif_transposed=transposed)


def from_array(rgb: np.ndarray, name: str, max_infer_side: int = 1920) -> PreparedImage:
    """Wrap an already-decoded RGB frame (e.g. video)."""
    h, w = rgb.shape[:2]
    scale, infer = 1.0, rgb
    if max(w, h) > max_infer_side:
        import cv2
        s = max(w, h) / max_infer_side
        infer = cv2.resize(rgb, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)
        scale = w / infer.shape[1]
    return PreparedImage(name=name, rgb=rgb, infer_rgb=infer, scale=scale, format=None, exif_transposed=False)
