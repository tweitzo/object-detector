
from functools import lru_cache

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

MODEL_ID = "facebook/detr-resnet-50"


@lru_cache(maxsize=1)
def load_detector():
    """Загружает модель один раз и повторно использует её."""
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="object-detection",
        model=MODEL_ID,
        device=device,
    )


def detect_objects(image: Image.Image, threshold: float = 0.70) -> list[dict]:
    """Находит объекты на изображении с заданным порогом уверенности."""
    detector = load_detector()
    detections = detector(image, threshold=threshold)

    return [
        item for item in detections
        if float(item["score"]) >= threshold
    ]


def color_for_label(label: str) -> tuple[int, int, int]:
    """Возвращает постоянный цвет для каждого класса."""
    value = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(label)
    )

    return (
        70 + value % 150,
        70 + (value * 3) % 150,
        70 + (value * 7) % 150,
    )


def annotate_image(
    image: Image.Image,
    detections: list[dict],
) -> Image.Image:
    """Рисует рамки и подписи найденных объектов."""
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()

    for item in detections:
        box = item["box"]
        label = item["label"]
        confidence = item["score"]
        color = color_for_label(label)

        x1 = int(box["xmin"])
        y1 = int(box["ymin"])
        x2 = int(box["xmax"])
        y2 = int(box["ymax"])

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=4,
        )

        caption = f"{label} {confidence:.0%}"
        text_position = (x1, max(0, y1 - 18))
        text_box = draw.textbbox(
            text_position,
            caption,
            font=font,
        )

        draw.rectangle(text_box, fill=color)
        draw.text(
            text_position,
            caption,
            fill="white",
            font=font,
        )

    return result
