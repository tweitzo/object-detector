
from collections import Counter
from io import BytesIO
from functools import lru_cache

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


MODEL_ID = "facebook/detr-resnet-50"
MAX_IMAGE_SIZE_MB = 15


st.set_page_config(
    page_title="Object Detector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Распознавание с картинки")
st.write(
    "Загрузите изображение, чтобы найти на нём объекты "
    "с помощью предварительно обученной модели компьютерного зрения."
)


@lru_cache(maxsize=1)
def load_detector():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="object-detection",
        model=MODEL_ID,
        device=device,
    )


def detect_objects(image, threshold):
    detector = load_detector()
    detections = detector(image, threshold=threshold)

    return [
        item for item in detections
        if float(item["score"]) >= threshold
    ]


def color_for_label(label):
    value = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(label)
    )

    return (
        70 + value % 150,
        70 + (value * 3) % 150,
        70 + (value * 7) % 150,
    )


def annotate_image(image, detections):
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


def main():
    with st.sidebar:
        st.header("Настройки")

        threshold = st.slider(
            "Порог уверенности",
            min_value=0.05,
            max_value=0.99,
            value=0.70,
            step=0.05,
        )

        st.caption(f"Модель: `{MODEL_ID}`")
        st.caption(
            "Модель обучена на COCO и распознаёт "
            "ограниченный набор повседневных объектов."
        )

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["jpg", "jpeg", "png", "webp"],
        help=f"Максимальный размер — {MAX_IMAGE_SIZE_MB} МБ.",
    )

    if uploaded_file is None:
        st.info("Загрузите изображение, чтобы начать распознавание.")
        return

    if uploaded_file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        st.error(
            f"Файл слишком большой. Максимальный размер — "
            f"{MAX_IMAGE_SIZE_MB} МБ."
        )
        return

    try:
        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
    except Exception:
        st.error("Не удалось открыть изображение.")
        return

    st.subheader("Исходное изображение")
    st.image(image, use_container_width=True)

    if st.button(
        "Распознать объекты",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Анализируем изображение…"):
                detections = detect_objects(image, threshold)
                annotated = annotate_image(image, detections)

            st.session_state["detections"] = detections
            st.session_state["annotated_image"] = annotated

        except Exception as error:
            st.error("Во время распознавания произошла ошибка.")
            st.exception(error)

    if "detections" not in st.session_state:
        return

    detections = st.session_state["detections"]
    annotated = st.session_state["annotated_image"]

    st.divider()
    st.subheader("Результат распознавания")

    left, right = st.columns(2)

    with left:
        st.image(
            annotated,
            caption="Обнаруженные объекты",
            use_container_width=True,
        )

    with right:
        st.metric("Всего объектов", len(detections))

        unique_objects = len({
            item["label"] for item in detections
        })

        st.metric("Уникальных классов", unique_objects)

        if detections:
            counts = Counter(
                item["label"] for item in detections
            )

            st.write("**Найденные классы:**")

            for label, count in counts.most_common():
                st.write(f"- {label}: {count}")
        else:
            st.warning(
                "Объекты не найдены. Попробуйте уменьшить порог."
            )

    if detections:
        st.subheader("Подробная информация")

        rows = []

        for index, item in enumerate(detections, start=1):
            box = item["box"]

            rows.append({
                "№": index,
                "Объект": item["label"],
                "Уверенность": f'{item["score"]:.1%}',
                "X min": round(box["xmin"]),
                "Y min": round(box["ymin"]),
                "X max": round(box["xmax"]),
                "Y max": round(box["ymax"]),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    output = BytesIO()
    annotated.save(output, format="PNG")

    st.download_button(
        "⬇️ Скачать изображение с рамками",
        data=output.getvalue(),
        file_name="detected_objects.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
