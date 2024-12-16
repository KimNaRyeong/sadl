import numpy as np
import os
import pickle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.datasets import mnist
from sa import get_ats  # 위에서 정의한 sa.py의 get_ats 함수 사용

# MNIST 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 60,000 / 10,000
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = Sequential([
    Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu', name="fc1"),
    Dropout(0.5),
    Dense(10, activation='softmax', name="fc2")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 모델 훈련
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.1)

# Fully Connected Layers의 이름을 정의
layer_names_fc1 = ["fc1"]
layer_names_fc2 = ["fc2"]

# Activation Traces 추출 및 저장 경로 설정
save_path = "./activation_traces/"
os.makedirs(save_path, exist_ok=True)

# 라벨별로 저장할 라벨 목록
target_labels = [0, 2, 3, 5, 6, 7, 8]

# 각 라벨별로 fc1 및 fc2의 Activation Traces 저장
for label in target_labels:
    # 해당 라벨의 인덱스 추출
    indices = np.where(y_test == label)[0]
    label_x_test = x_test[indices]

    # fc1 (128차원) Activation Traces 계산
    ats_fc1, _ = get_ats(
        model=model,
        dataset=label_x_test,
        name=f"mnist_fc1_label_{label}",
        layer_names=layer_names_fc1,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10
    )

    # fc2 (10차원) Activation Traces 계산
    ats_fc2, _ = get_ats(
        model=model,
        dataset=label_x_test,
        name=f"mnist_fc2_label_{label}",
        layer_names=layer_names_fc2,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10
    )

    # 저장 경로 정의
    ats_fc1_path = os.path.join(save_path, f"mnist_fc1_ats_label_{label}.pkl")
    ats_fc2_path = os.path.join(save_path, f"mnist_fc2_ats_label_{label}.pkl")

    # Activation Traces를 pkl 파일로 저장
    with open(ats_fc1_path, "wb") as f:
        pickle.dump(ats_fc1, f)

    with open(ats_fc2_path, "wb") as f:
        pickle.dump(ats_fc2, f)

    print(f"Activation Traces for fc1 (label {label}) saved to: {ats_fc1_path}")
    print(f"Activation Traces for fc2 (label {label}) saved to: {ats_fc2_path}")
