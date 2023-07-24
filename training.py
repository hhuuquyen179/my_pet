import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Định nghĩa đường dẫn đến các thư mục chứa dữ liệu
happy_folder = "/Users/Admin/Documents/GitHub/my_pet/happy"
sad_folder = "/Users/Admin/Documents/GitHub/my_pet/Sad"
angry_folder = "/Users/Admin/Documents/GitHub/my_pet/Angry"
other_folder = "/Users/Admin/Documents/GitHub/my_pet/Other"

# Hàm để tải và tiền xử lý ảnh
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))  # Điều chỉnh kích thước ảnh thành 48x48 điểm ảnh
            images.append(img)
    return images

# Tải dữ liệu ảnh và nhãn cho mỗi cảm xúc
happy_images = load_images_from_folder(happy_folder)
sad_images = load_images_from_folder(sad_folder)
angry_images = load_images_from_folder(angry_folder)
other_images = load_images_from_folder(other_folder)

# Tạo nhãn cho mỗi cảm xúc
happy_labels = [0] * len(happy_images)
sad_labels = [1] * len(sad_images)
angry_labels = [2] * len(angry_images)
other_labels = [3] * len(other_images)

# Gộp ảnh và nhãn vào mảng numpy
X = np.array(happy_images + sad_images + angry_images + other_images)
y = np.array(happy_labels + sad_labels + angry_labels + other_labels)

# Chuẩn hóa giá trị pixel về khoảng [0, 1]
X = X.astype('float32') / 255.0

# One-hot encode nhãn
y = tf.keras.utils.to_categorical(y, num_classes=4)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train.reshape(-1, 48, 48, 1), y_train, batch_size=32, epochs=20, validation_split=0.1)

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test.reshape(-1, 48, 48, 1), y_test)
print(f"Độ chính xác kiểm tra: {accuracy*100:.2f}%")

# Lưu mô hình đã huấn luyện
model.save("facial_expression_model.h5")
