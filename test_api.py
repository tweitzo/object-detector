from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Object Detector API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_invalid_file():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_detect_valid_image(monkeypatch):
    def fake_detect_objects(image, threshold):
        return [
            {
                "score": 0.95,
                "label": "person",
                "box": {
                    "xmin": 10,
                    "ymin": 20,
                    "xmax": 100,
                    "ymax": 200,
                },
            }
        ]

    monkeypatch.setattr("api.detect_objects", fake_detect_objects)

    image = Image.new("RGB", (100, 100), color="white")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")

    response = client.post(
        "/detect",
        files={
            "file": (
                "test.jpg",
                image_bytes.getvalue(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["total_objects"] == 1
    assert response.json()["detections"][0]["label"] == "person"