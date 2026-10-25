"""
================================================================================
KODE PROGRAM PENELITIAN SKRIPSI (VERSI v1.0)
Judul   : Klasifikasi Multi-Kelas Tumor Otak Citra MRI Menggunakan ResNet50
          dengan Interpretasi Grad-CAM
Peneliti: Aw. Avril Rindra Pratama (2311102441032)
Prodi   : S1 Teknik Informatika - Universitas Muhammadiyah Kalimantan Timur
Repo    : https://github.com/AwAvrilRindraPratama/DataSetMRI.git
================================================================================
"""

import os
import json
import random
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)

# ==============================================================================
# 1. KONFIGURASI PARAMETER & PENGUNCI KEACAKAN (TABEL II.4 PROPOSAL)
# ==============================================================================
SEED = 42


def set_global_seed(seed=SEED):
    """Mengunci seluruh generator angka acak agar eksperimen 100% dapat direproduksi."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seed(SEED)

# Deteksi otomatis lokasi folder dataset (mendukung eksekusi lokal, GitHub repo, & Colab)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "dataset_1600"),
    os.path.abspath(os.path.join(_SCRIPT_DIR, "..")),
    r"D:\Metopen\dataset_1600",
    "/content/dataset_1600",
]
DATASET_DIR = next(
    (
        p
        for p in _DEFAULT_DATASET_CANDIDATES
        if os.path.isdir(os.path.join(p, "Training"))
        and os.path.isdir(os.path.join(p, "Testing"))
    ),
    r"D:\Metopen\dataset_1600",
)
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "outputs_5fold")

IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
N_SPLITS = 5
DROPOUT_RATE = 0.4
LR_FREEZE = 1e-4
LR_FINETUNE = 1e-5
EPOCHS_WARMUP = 15
EPOCHS_FINETUNE = 15
EPOCHS_FULL_FREEZE = 30
TARGET_LAYER = "conv5_block3_out"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)


# ==============================================================================
# 2. PENGUMPULAN DATA & PARTISI STRATIFIED 5-FOLD (SUBBAB II.3.1)
# ==============================================================================
def collect_dataset(dataset_dir, classes):
    """Mengumpulkan 1.600 path berkas citra (.jpg) dan label kelasnya (0..3)."""
    filepaths, labels = [], []
    for split_folder in ["Training", "Testing"]:
        split_path = os.path.join(dataset_dir, split_folder)
        if not os.path.isdir(split_path):
            continue
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in sorted(os.listdir(class_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepaths.append(os.path.join(class_path, fname))
                    labels.append(class_idx)
    return np.array(filepaths), np.array(labels, dtype=np.int32)


# ==============================================================================
# 3. PERSIAPAN DATA & AUGMENTASI GEOMETRIS RINGAN (SUBBAB II.3.2)
# ==============================================================================
# Rotasi +-10 derajat setara dengan factor = 10 / 360 = 0.0278 (~0.028)
mild_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(factor=0.028, fill_mode="nearest", seed=SEED),
    ],
    name="mild_geometric_augmentation",
)


def parse_and_preprocess(filepath, label, training=False):
    """Membaca berkas JPEG, resize bilinear 224x224, augmentasi latih, & normalisasi ImageNet."""
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    if training:
        img = mild_augmentation(tf.expand_dims(img, axis=0), training=True)[0]

    img = preprocess_input(img)
    label_onehot = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label_onehot


def build_tf_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, y: parse_and_preprocess(p, y, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 4. PERANCANGAN MODEL RESNET50 & PARTIAL FINE-TUNING (SUBBAB II.3.3 & KODE II.1)
# ==============================================================================
def build_resnet50_classifier():
    """Membangun model fungsional ResNet50 (Full-Freeze / Warm-up)."""
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(DROPOUT_RATE, seed=SEED, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(
        inputs=base_model.input, outputs=outputs, name="ResNet50_BrainTumor_4Class"
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_FREEZE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_conv5_block3(model):
    """Membuka pembekuan blok 'conv5_block3' untuk Skenario Partial Fine-Tuning."""
    for layer in model.layers:
        if layer.name.startswith("conv5_block3"):
            layer.trainable = not isinstance(layer, layers.BatchNormalization)
        elif layer.name in ["gap", "dropout", "predictions"]:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks():
    """Membuat instance baru EarlyStopping & ReduceLROnPlateau untuk setiap fase fit()."""
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]


# ==============================================================================
# 5. EKSTRAKSI VISUALISASI GRAD-CAM (SUBBAB II.3.4 - PERSAMAAN 1 & 2)
# ==============================================================================
def extract_gradcam(img_preprocessed, model, target_layer=TARGET_LAYER, class_idx=None):
    """Mengekstrak peta aktivasi Grad-CAM dari layer 'conv5_block3_out'."""
    conv_layer = model.get_layer(target_layer)
    grad_model = models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_preprocessed, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        target_score = predictions[:, class_idx]

    # Persamaan (1): Bobot kepentingan kanal alpha_k^c
    grads = tape.gradient(target_score, conv_outputs)
    alpha_kc = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Persamaan (2): Kombinasi linear berbobot + ReLU
    feature_maps = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(feature_maps, alpha_kc), axis=-1)
    cam = tf.maximum(cam, 0.0)

    cam_np = cam.numpy()
    cam_max = np.max(cam_np)
    if cam_max > 0:
        cam_np = cam_np / cam_max
    heatmap_224 = cv2.resize(cam_np, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    prob_score = float(predictions[0, class_idx].numpy())

    return heatmap_224, class_idx, prob_score


def create_gradcam_overlay(rgb_img_224, heatmap_224, alpha=0.4):
    """Menumpangkan peta panas warna JET di atas citra MRI asli."""
    heatmap_uint8 = np.uint8(255 * heatmap_224)
    heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        rgb_img_224.astype(np.uint8), 1.0 - alpha, heatmap_jet_rgb, alpha, 0
    )
    return overlay


# ==============================================================================
# 6. PENGUJIAN KUANTITATIF FAITHFULNESS XAI (SUBBAB II.3.5 - PERSAMAAN 9 & 10)
# ==============================================================================
def compute_xai_faithfulness_metrics(model, test_paths, target_layer=TARGET_LAYER):
    """Menghitung Average Drop (%) dan Increase in Confidence (%) pada data uji."""
    drop_values = []
    confidence_increases = []

    for path in test_paths:
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(
            cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        img_float = img_rgb.astype(np.float32)

        # Skor keyakinan pada citra asli utuh (Y_n^c)
        inp_full = preprocess_input(np.expand_dims(img_float.copy(), axis=0))
        heatmap_224, pred_class, y_nc = extract_gradcam(
            inp_full, model, target_layer=target_layer
        )

        # Citra penjelasan termasker: E_n = I_n * L_GradCAM
        explanation_img = img_float * np.expand_dims(heatmap_224, axis=-1)
        inp_masked = preprocess_input(np.expand_dims(explanation_img.copy(), axis=0))

        # Skor keyakinan pada citra termasker (O_n^c)
        preds_masked = model(inp_masked, training=False).numpy()
        o_nc = float(preds_masked[0, pred_class])

        # Persamaan (9): Average Drop (%)
        drop_n = max(0.0, y_nc - o_nc) / max(y_nc, 1e-8)
        drop_values.append(drop_n)

        # Persamaan (10): Increase in Confidence (%)
        confidence_increases.append(1.0 if o_nc > y_nc else 0.0)

    average_drop_pct = float(np.mean(drop_values) * 100.0)
    increase_conf_pct = float(np.mean(confidence_increases) * 100.0)
    return average_drop_pct, increase_conf_pct


# ==============================================================================
# 7. EKSEKUSI LINTAS-LIPATAN & VISUALISASI HASIL (SUBBAB II.3.5 & II.3.6)
# ==============================================================================
def save_fold_visualizations(
    fold_idx, scenario, test_labels, y_preds, y_probs, model, test_paths, output_dir
):
    """Menyimpan plot Confusion Matrix, Kurva ROC-AUC, dan Panel Grad-CAM."""
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(test_labels, y_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {scenario} (Fold {fold_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{scenario}_fold{fold_idx}.png"), dpi=300)
    plt.close()

    y_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)
    plt.figure(figsize=(6.5, 5))
    for i, cls_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        cls_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, label=f"{cls_name} (AUC = {cls_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"Kurva ROC One-vs-Rest - {scenario} (Fold {fold_idx})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_{scenario}_fold{fold_idx}.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(4, 3, figsize=(9, 11))
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_indices = np.where(test_labels == cls_idx)[0]
        sample_path = test_paths[cls_indices[0]]

        img_bgr = cv2.imread(sample_path)
        img_rgb = cv2.cvtColor(
            cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        inp = preprocess_input(np.expand_dims(img_rgb.astype(np.float32).copy(), axis=0))
        heatmap, pred_idx, conf = extract_gradcam(inp, model)
        overlay = create_gradcam_overlay(img_rgb, heatmap, alpha=0.4)

        axes[cls_idx, 0].imshow(img_rgb)
        axes[cls_idx, 0].set_title(f"Citra Asli: {cls_name}")
        axes[cls_idx, 0].axis("off")

        axes[cls_idx, 1].imshow(heatmap, cmap="jet")
        axes[cls_idx, 1].set_title("Peta Aktivasi Grad-CAM")
        axes[cls_idx, 1].axis("off")

        axes[cls_idx, 2].imshow(overlay)
        axes[cls_idx, 2].set_title(f"Pred: {CLASSES[pred_idx]} ({conf*100:.1f}%)")
        axes[cls_idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"gradcam_{scenario}_fold{fold_idx}.png"), dpi=300
    )
    plt.close()


def run_experiment(
    scenario="partial_finetune", dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR
):
    """Menjalankan Stratified 5-Fold CV untuk 'full_freeze' atau 'partial_finetune'."""
    os.makedirs(output_dir, exist_ok=True)
    filepaths, labels = collect_dataset(dataset_dir, CLASSES)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(filepaths, labels), start=1
    ):
        print(
            f"\n==================== FOLD {fold_idx}/{N_SPLITS} [{scenario}] ===================="
        )
        set_global_seed(SEED + fold_idx)
        train_paths, train_labels = filepaths[train_idx], labels[train_idx]
        test_paths, test_labels = filepaths[test_idx], labels[test_idx]

        train_ds = build_tf_dataset(train_paths, train_labels, training=True)
        test_ds = build_tf_dataset(test_paths, test_labels, training=False)

        model = build_resnet50_classifier()

        if scenario == "full_freeze":
            model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=EPOCHS_FULL_FREEZE,
                callbacks=build_callbacks(),
                verbose=1,
            )
        elif scenario == "partial_finetune":
            model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=EPOCHS_WARMUP,
                callbacks=build_callbacks(),
                verbose=1,
            )
            model = unfreeze_conv5_block3(model)
            model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=EPOCHS_FINETUNE,
                callbacks=build_callbacks(),
                verbose=1,
            )

        # Evaluasi Klasifikasi (Persamaan 3 s.d. 6)
        y_probs = model.predict(test_ds, verbose=0)
        y_preds = np.argmax(y_probs, axis=1)
        y_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

        acc = accuracy_score(test_labels, y_preds) * 100.0
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            test_labels, y_preds, average="macro", zero_division=0
        )
        macro_auc = roc_auc_score(
            y_onehot, y_probs, multi_class="ovr", average="macro"
        )

        # Evaluasi Faithfulness Grad-CAM (Persamaan 9 & 10)
        avg_drop, inc_conf = compute_xai_faithfulness_metrics(model, test_paths)

        save_fold_visualizations(
            fold_idx,
            scenario,
            test_labels,
            y_preds,
            y_probs,
            model,
            test_paths,
            output_dir,
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "accuracy_pct": float(acc),
                "macro_precision": float(macro_p),
                "macro_recall": float(macro_r),
                "macro_f1": float(macro_f1),
                "roc_auc_ovr": float(macro_auc),
                "average_drop_pct": float(avg_drop),
                "increase_in_confidence_pct": float(inc_conf),
            }
        )

        # Bersihkan sesi graf di akhir setiap lipatan agar memori GPU tidak bocor
        tf.keras.backend.clear_session()

    # Ringkasan Rata-rata & Simpangan Baku 5-Fold (Persamaan 7 & 8)
    aggregate = {}
    for metric_name in [
        "accuracy_pct",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc_ovr",
        "average_drop_pct",
        "increase_in_confidence_pct",
    ]:
        values = [f[metric_name] for f in fold_results]
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    report_path = os.path.join(output_dir, f"hasil_evaluasi_5fold_{scenario}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"folds": fold_results, "aggregate_mean_std": aggregate}, f, indent=2
        )

    return aggregate


def print_comparison_summary(res_freeze, res_finetune):
    """Mencetak tabel komparasi akhir (Mean +- Std) antara kedua skenario."""
    print("\n" + "=" * 78)
    print(f"{'METRIK EVALUASI (5-FOLD CV)':32s} | {'FULL-FREEZE':20s} | {'PARTIAL FINE-TUNE':20s}")
    print("-" * 78)
    for k in res_freeze:
        m1, s1 = res_freeze[k]["mean"], res_freeze[k]["std"]
        m2, s2 = res_finetune[k]["mean"], res_finetune[k]["std"]
        print(f"{k:32s} | {m1:8.4f} ± {s1:7.4f} | {m2:8.4f} ± {s2:7.4f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eksperimen 5-Fold CV ResNet50 + Grad-CAM untuk Klasifikasi MRI Tumor Otak"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DATASET_DIR,
        help="Path menuju direktori dataset_1600",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Path direktori keluaran hasil evaluasi dan plot",
    )
    args = parser.parse_args()

    paths, lbls = collect_dataset(args.dataset_dir, CLASSES)
    print(f"Dataset siap: {len(paths)} citra ({len(CLASSES)} kelas seimbang).")

    hasil_freeze = run_experiment(
        scenario="full_freeze",
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    hasil_finetune = run_experiment(
        scenario="partial_finetune",
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    print_comparison_summary(hasil_freeze, hasil_finetune)
