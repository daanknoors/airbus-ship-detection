import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from airbus_ship_detection.main import app
from airbus_ship_detection import configs
import io

client = TestClient(app)

def test_get_ship_mask():
    # load random image
    img = configs.DIR_DATA_TRAIN_IMG / "0a0df8299.jpg"
    image = Image.open(img)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("test.jpg", buf, "image/jpeg")}
    response = client.post("/image/mask", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    mask = Image.open(io.BytesIO(response.content))
    mask_np = np.array(mask)
    assert mask_np.ndim == 2  # binary mask
    assert mask_np.shape == (768, 768)
    assert set(np.unique(mask_np)).issubset({0, 255})  #
    assert np.sum(mask_np) > 0  # at least some ship pixels
    assert np.sum(mask_np) < 768 * 768  # not all ship pixels
    buf.close()
