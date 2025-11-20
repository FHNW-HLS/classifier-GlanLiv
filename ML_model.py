import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# ---------------- Configuration ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 32
NUM_FOLDS = 5

DATA_PATHS = {
    'open': 'class_open',
    'closed': 'class_closed'
}
MODEL_SAVE_PATH = 'best_edge_classifier_model.h5'
PLOTS_DIR = 'training_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- Utilities ----------------
def load_images_from_folder(folder, label, img_size):
    images, labels = [], []
    if not os.path.isdir(folder):
        print(f"⚠️ Folder not found: {folder}")
        return images, labels

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L')
                    img = img.resize((img_size, img_size))
                    img_np = np.array(img, dtype=np.float32)
                    images.append(img_np)
                    labels.append(label)
            except Exception as e:
                print(f"⚠️ Failed to load {img_path}: {e}")
    return images, labels

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

def plot_and_save_history(history, fold, outdir=PLOTS_DIR):
    hist = history.history
    epochs_range = range(1, len(hist.get('loss', [])) + 1)
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']

    for metric in metrics:
        if metric in hist:
            plt.figure()
            plt.plot(epochs_range, hist.get(metric, []), label=f'train_{metric}')
            val_metric = f'val_{metric}'
            if val_metric in hist:
                plt.plot(epochs_range, hist.get(val_metric, []), label=f'val_{metric}')
            plt.title(f'Fold {fold} - {metric}')
            plt.legend()
            path = os.path.join(outdir, f'fold_{fold}_{metric}.png')
            plt.savefig(path)
            plt.close()
            print(f"Saved {path}")

# ---------------- Load dataset ----------------
images_open, labels_open   = load_images_from_folder(DATA_PATHS['open'],   label=1, img_size=IMG_SIZE)
images_closed, labels_closed = load_images_from_folder(DATA_PATHS['closed'], label=0, img_size=IMG_SIZE)

if len(images_open) + len(images_closed) == 0:
    raise SystemExit("No images found in specified data folders.")

X = np.array(images_open + images_closed, dtype=np.float32) / 255.0
y = np.array(labels_open + labels_closed, dtype=np.int32)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Total samples: {len(X)} (open={sum(y==1)}, closed={sum(y==0)})")

# ---- Shuffle BEFORE K-Fold ----
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# ---------------- Stratified K-Fold ----------------
kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

fold = 1
accuracies, aucs, precisions, recalls, f1s = [], [], [], [], []
best_val_auc = -1.0
overall_y_true, overall_y_pred = [], []

for train_idx, val_idx in kf.split(X, y):

    print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Class weights
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: cw_val for i, cw_val in enumerate(cw)}
    print(f"Class weights: {class_weights}")

    # Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.8,1.2],
        fill_mode='nearest'
    )
    datagen.fit(X_train, seed=SEED)

    # Model
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    # Callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, verbose=1, mode='max')

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # Predict
    y_val_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    overall_y_true.extend(y_val)
    overall_y_pred.extend(y_val_pred)

    # Metrics
    fold_acc = np.mean(y_val_pred == y_val)
    fold_auc = tf.keras.metrics.AUC()(y_val, y_val_prob).numpy()
    fold_precision = tf.keras.metrics.Precision()(y_val, y_val_pred).numpy()
    fold_recall = tf.keras.metrics.Recall()(y_val, y_val_pred).numpy()
    fold_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    accuracies.append(fold_acc)
    aucs.append(fold_auc)
    precisions.append(fold_precision)
    recalls.append(fold_recall)
    f1s.append(fold_f1)

    print(f"Fold {fold} results: Acc={fold_acc:.4f}, AUC={fold_auc:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}, F1={fold_f1:.4f}")

    # Save best model
    if fold_auc > best_val_auc:
        best_val_auc = fold_auc
        model.save(MODEL_SAVE_PATH)
        print(f"✅ New best model saved to: {MODEL_SAVE_PATH}")

        cm = confusion_matrix(y_val, y_val_pred)
        cr = classification_report(y_val, y_val_pred, target_names=['Closed', 'Open'])
        np.save(os.path.join(PLOTS_DIR, 'best_fold_confusion_matrix.npy'), cm)
        with open(os.path.join(PLOTS_DIR, 'best_fold_classification_report.txt'), 'w') as f:
            f.write(cr)

    # Save fold data
    cm_fold = confusion_matrix(y_val, y_val_pred)
    np.save(os.path.join(PLOTS_DIR, f'fold_{fold}_confusion_matrix.npy'), cm_fold)
    with open(os.path.join(PLOTS_DIR, f'fold_{fold}_classification_report.txt'), 'w') as f:
        f.write(classification_report(y_val, y_val_pred, target_names=['Closed', 'Open']))

    plot_and_save_history(history, fold)

    K.clear_session()
    fold += 1

# ---------------- Final summary ----------------
overall_y_true = np.array(overall_y_true)
overall_y_pred = np.array(overall_y_pred)
overall_cm = confusion_matrix(overall_y_true, overall_y_pred)
overall_cr = classification_report(overall_y_true, overall_y_pred, target_names=['Closed', 'Open'])
overall_f1 = f1_score(overall_y_true, overall_y_pred, zero_division=0)

print("\n=== Overall Results ===")
print("Confusion Matrix:\n", overall_cm)
print("\nClassification Report:\n", overall_cr)
print(f"Overall F1: {overall_f1:.4f}")

np.save(os.path.join(PLOTS_DIR, 'overall_confusion_matrix.npy'), overall_cm)
with open(os.path.join(PLOTS_DIR, 'overall_classification_report.txt'), 'w') as f:
    f.write(overall_cr)

print(f"\nBest model saved at: {MODEL_SAVE_PATH}")
print("Done.")
