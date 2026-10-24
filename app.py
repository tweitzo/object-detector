from collections import Counter
from io import BytesIO

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

st.title("🔎 Распознование с картинки")
st.write(
    "Загрузите изображение, чтобы найти на нём объекты с помощью "
    "предварительно обученной модели компьютерного зрения."
)


@st.cache_resource(show_spinner="Загружаем нейросеть Hugging Face…")
def load_detector():
    """Load the pretrained object-detection model once per app session."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="object-detection",
        model=MODEL_ID,
        device=device,
    )


def color_for_label(label: str) -> tuple[int, int, int]:
    """Return a stable, readable color for each detected class."""
    value = sum((index + 1) * ord(char) for index, char in enumerate(label))
    return (70 + value % 150, 70 + (value * 3) % 150, 70 + (value * 7) % 150)


def annotate_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw detection boxes and labels on a copy of the uploaded image."""
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

        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        caption = f"{label} {confidence:.0%}"
        text_box = draw.textbbox((x1, max(0, y1 - 18)), caption, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, max(0, y1 - 18)), caption, fill="white", font=font)

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
            help="Показывать только объекты, уверенность модели в которых "
                 "не ниже выбранного значения.",
        )
        st.caption(f"Модель: `{MODEL_ID}`")
        st.caption(
            "Модель обучена на COCO и распознаёт ограниченный набор "
            "повседневных категорий объектов."
        )

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["jpg", "jpeg", "png", "webp"],
        help=f"Поддерживаются JPG, PNG и WEBP. Максимальный размер — "
             f"{MAX_IMAGE_SIZE_MB} МБ.",
    )

    if uploaded_file is None:
        st.info("Загрузите изображение, чтобы начать распознавание.")
        st.markdown(
            "**Как это работает:** изображение передаётся модели DETR, "
            "которая определяет классы объектов и координаты рамок."
        )
        return

    if uploaded_file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        st.error(f"Файл слишком большой. Максимальный размер — {MAX_IMAGE_SIZE_MB} МБ.")
        return

    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception:
        st.error("Не удалось открыть файл. Проверьте, что это корректное изображение.")
        return

    st.subheader("Исходное изображение")
    st.image(image, use_container_width=True)

    if st.button("Распознать объекты", type="primary", use_container_width=True):
        try:
            detector = load_detector()
            with st.spinner("Анализируем изображение…"):
                raw_detections = detector(image, threshold=threshold)

            detections = [
                item for item in raw_detections
                if float(item["score"]) >= threshold
            ]
            annotated = annotate_image(image, detections)

            st.session_state["detections"] = detections
            st.session_state["annotated_image"] = annotated
            st.session_state["source_name"] = uploaded_file.name
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
        st.image(annotated, caption="Обнаруженные объекты", use_container_width=True)
    with right:
        st.metric("Всего объектов", len(detections))
        unique_objects = len({item["label"] for item in detections})
        st.metric("Уникальных классов", unique_objects)

        if detections:
            counts = Counter(item["label"] for item in detections)
            st.write("**Найденные классы:**")
            for label, count in counts.most_common():
                st.write(f"- {label}: {count}")
        else:
            st.warning(
                "Объекты с выбранным порогом уверенности не найдены. "
                "Попробуйте уменьшить порог."
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
