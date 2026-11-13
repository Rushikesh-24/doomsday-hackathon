import io

import numpy as np
import pytest
from PIL import Image

from tests.conftest import png_bytes, to_bytes
from vision.preprocessing.image import InvalidImageError, load_image


def test_valid_jpeg():
    img = load_image(to_bytes(np.zeros((100, 200, 3), np.uint8)), "a.jpg")
    assert (img.width, img.height) == (200, 100) and img.format == "JPEG" and img.scale == 1.0


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"not an image at all", "not a decodable"),
                                      (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "not a decodable")])
def test_invalid_bytes(data, msg):
    with pytest.raises(InvalidImageError, match=msg):
        load_image(data, "bad")


def test_truncated_jpeg():
    data = to_bytes(np.random.randint(0, 255, (300, 300, 3), np.uint8))
    with pytest.raises(InvalidImageError):
        load_image(data[: len(data) // 3], "trunc.jpg")


def test_too_small_and_too_big():
    with pytest.raises(InvalidImageError, match="too small"):
        load_image(png_bytes(np.zeros((10, 10, 3), np.uint8)), "tiny.png")
    with pytest.raises(InvalidImageError, match="too large"):
        load_image(png_bytes(np.zeros((40, 100, 3), np.uint8)), "wide.png", max_side=80)


def test_file_size_limit():
    with pytest.raises(InvalidImageError, match="file too large"):
        load_image(png_bytes(np.zeros((64, 64, 3), np.uint8)), "x.png", max_bytes=10)


def test_unsupported_format():
    b = io.BytesIO()
    Image.new("RGB", (64, 64)).save(b, format="GIF")
    with pytest.raises(InvalidImageError, match="unsupported format"):
        load_image(b.getvalue(), "x.gif")


def test_rgba_and_grayscale_converted():
    rgba = np.zeros((50, 50, 4), np.uint8)
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, format="PNG")
    img = load_image(b.getvalue(), "a.png")
    assert img.rgb.shape == (50, 50, 3) and img.rgb[0, 0].tolist() == [255, 255, 255]  # transparent -> white
    g = io.BytesIO()
    Image.new("L", (40, 40), 7).save(g, format="PNG")
    assert load_image(g.getvalue(), "g.png").rgb.shape == (40, 40, 3)


def test_exif_orientation_applied():
    arr = np.zeros((100, 200, 3), np.uint8)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    b = io.BytesIO()
    im.save(b, format="JPEG", exif=exif)
    img = load_image(b.getvalue(), "rot.jpg")
    assert (img.width, img.height) == (100, 200) and img.exif_transposed


def test_downscale_for_inference():
    img = load_image(to_bytes(np.zeros((1000, 4000, 3), np.uint8)), "big.jpg", max_infer_side=1000)
    assert img.infer_rgb.shape[1] == 1000 and img.width == 4000 and abs(img.scale - 4.0) < 1e-6
