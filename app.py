import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Hàm để tải và tiền xử lý ảnh
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48))
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_expanded, axis=-1)
    return img_final

# Tải mô hình đã huấn luyện
model = load_model("facial_expression_model.h5")

# Hàm để dự đoán cảm xúc trên ảnh đầu vào
def predict_emotion(image):
    img_final = preprocess_image(image)
    prediction = model.predict(img_final)
    emotion_classes = {0: 'vui vẻ', 1: 'buồn', 2: 'giận dữ', 3: 'khác'}
    emotion_label = np.argmax(prediction[0])
    predicted_emotion = emotion_classes[emotion_label]
    return predicted_emotion, prediction[0]

# Ứng dụng web Streamlit
st.title("Ứng dụng Nhận diện Cảm xúc Khuôn mặt")
st.write("Tải lên một ảnh và ứng dụng sẽ dự đoán cảm xúc trên khuôn mặt.")

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Chuyển đổi ảnh từ định dạng PIL sang OpenCV
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if st.button("Dự đoán"):
        predicted_emotion, prediction_prob = predict_emotion(image_cv2)
        st.write(f"Cảm xúc dự đoán: {predicted_emotion}")
        st.write(f"Xác suất [vui vẻ, buồn, giận dữ, khác]: {prediction_prob}")

        # Hiển thị ảnh thử nghiệm cùng với nhãn cảm xúc dự đoán
        st.image(image, caption=f"Cảm xúc dự đoán: {predicted_emotion}", use_column_width=True)
