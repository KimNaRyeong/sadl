import pickle, os

# 저장된 pkl 파일들의 차원을 확인
save_path = "./activation_traces/"
labels_to_save = range(10)
for label in labels_to_save:
    ats_label_path_fc1 = os.path.join(save_path, f"mnist_fc1_ats_label_{label}.pkl")
    ats_label_path_fc2 = os.path.join(save_path, f"mnist_fc2_ats_label_{label}.pkl")
    # pred_label_path = os.path.join(save_path, f"mnist_fc_pred_label_{label}.pkl")

    # Activation Traces 로드 및 차원 출력
    with open(ats_label_path_fc1, "rb") as f:
        ats_label = pickle.load(f)
        print(f"fc1 Activation Traces for label {label}: {ats_label.shape}")

    # Predictions 로드 및 차원 출력
    with open(ats_label_path_fc2, "rb") as f:
        ats_label = pickle.load(f)
        print(f"fc2 Predictions for label {label}: {ats_label.shape}")

ats_path = os.path.join(save_path, "mnist_fc_ats.pkl")
pred_path = os.path.join(save_path, "mnist_fc_pred.pkl")

with open(ats_path, 'rb') as f:
    ats = pickle.load(f)
    print(f"Activation Traces: {ats.shape}")

with open(pred_path, 'rb') as f:
    pred = pickle.load(f)
    print(f"Predictions: {pred.shape}")
