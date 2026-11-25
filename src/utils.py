from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
import numpy as np

def load_data(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        df = pd.read_csv(path)
        print("Dataset loaded.")
        return df
    else:
        print("Dataset not found.")
        return None

def class_distribution(
    df: pd.DataFrame,
    normalize: bool = False,
    name: str = ""
) -> None:
    counts_class = df['dx'].value_counts(normalize=normalize)

    plt.figure(figsize=(10, 5))
    bars = counts_class.plot(kind='bar')

    fmt = '%.3f' if normalize else '%d'
    plt.bar_label(bars.containers[0], fmt=fmt, padding=1)

    plt.title('Class distribution ' + name)
    plt.xlabel('Class (dx)')
    plt.ylabel('Number of image samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def build_confusion_matrix(model, dataset, class_names):
    y_pred = model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([y.numpy() for x, y in dataset])
    y_true = np.argmax(y_true, axis=1)
    
    
    cmatrix = confusion_matrix(y_true, y_pred, normalize='true')
    
    display = ConfusionMatrixDisplay(
        confusion_matrix=cmatrix,
        display_labels=class_names
    )

    return cmatrix, display

def history_plot(history):

    if isinstance(history, pd.DataFrame):
        history = history.to_dict(orient="list")
    else:
        history = history.history

    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.plot(history['f1_macro'], label="F1 macro (train)")
    plt.plot(history['val_f1_macro'], label = "F1 macro (val)")
    plt.title("F1 macro")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(history['loss'], label="loss (train)")
    plt.plot(history['val_loss'], label = "loss (val)")
    plt.title("Loss function")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(history['accuracy'], label="accuracy (train)")
    plt.plot(history['val_accuracy'], label = "accuracy (val)")
    plt.title("accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_aug(augmentation=None, sample_batch=None, class_names=None):
    if sample_batch is None or class_names is None:
        print("Error")
        return
    fig, axes = plt.subplots(8,8, figsize=(15, 15))
    row, col = 0,0
    aug = augmentation()
    
    for sample, label in zip(*sample_batch):
        if col != 0 and col % 8 == 0:
            col=0
            row+=1
        ax = axes[row, col]
        ax.imshow(sample.astype('uint8'))
        ax.set_title(f"dx: {class_names[np.argmax(label)]}\n")
        ax.axis('off')
    
        img_aug = aug(sample)
        ax_aug = axes[row, col+1]
        ax_aug.imshow(img_aug.numpy().astype("uint8"))
        ax_aug.set_title(f"dx: {class_names[np.argmax(label)]} (after aug)\n")
        ax_aug.axis('off')
        col += 2
    
    plt.tight_layout()
    plt.show()

import tensorflow as tf
from tensorflow.keras import layers


def build_ft3_model(
    baseline_path,
    input_shape=(224, 224, 3),
    learning_rate=1e-5,
    unfreeze_blocks=(6, 7),
    augmentation=None,
    model_name="EfficientNetB0_FT3",
):

    base = tf.keras.models.load_model(
        baseline_path,
        compile=False
    )

    backbone = base.get_layer("efficientnetb0")
    backbone.trainable = True

    prefixes = tuple(
        f"block{i}" for i in unfreeze_blocks
    )

    for layer in backbone.layers:
        is_selected = (
            layer.name.startswith(prefixes)
            or layer.name.startswith("top_")
        )

        layer.trainable = (
            is_selected
            and not isinstance(
                layer, layers.BatchNormalization
            )
        )

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if augmentation is not None:
        x = augmentation(x)

    outputs = base(x)
    
    model = tf.keras.Model(
        inputs,
        outputs,
        name=model_name
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.F1Score(
                average="macro",
                name="f1_macro"
            )
        ]
    )

    return model

def evaluate_per_class(model, dataset, class_names):
    y_true = np.concatenate([
        np.argmax(y.numpy(), axis=1)
        for _, y in dataset
    ])

    y_pred = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).T.loc[
        class_names,
        ["precision", "recall", "f1-score", "support"]
    ]

def build_augmentation(experiment_id, SEED=42):
    layers = [tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=SEED)]
    if experiment_id == 'AUG1':
        layers += [tf.keras.layers.RandomRotation(20 / 360, fill_mode='reflect', seed=SEED),
                   tf.keras.layers.RandomZoom(0.05, fill_mode='reflect', seed=SEED)]
    elif experiment_id == 'AUG2':
        layers += [tf.keras.layers.RandomRotation(90 / 360, fill_mode='reflect', seed=SEED),
                   tf.keras.layers.RandomZoom(0.10, fill_mode='reflect', seed=SEED),
                   tf.keras.layers.RandomContrast(0.10, seed=SEED)]
    elif experiment_id == 'AUG3':
        layers += [tf.keras.layers.RandomRotation(0.5, fill_mode='reflect', seed=SEED),
                   tf.keras.layers.RandomZoom(0.15, fill_mode='reflect', seed=SEED),
                   tf.keras.layers.RandomContrast(0.20, seed=SEED),
                   tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255), seed=SEED)]
    else:
        return None
    return tf.keras.Sequential(layers, name=f'augmentation_{experiment_id}')