import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

import loader

def get_model(width, height, depth, drop = .3):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(drop)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def visualize_performance(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])
    if not os.path.isdir('figures'):
        os.mkdir('figures')
    plt.tight_layout()
    plt.savefig(f"figures/{name}_performance.svg")

    res = {'train_accuracy' : history.history['acc'], 'val_accuracy' : history.history['val_acc'],
            'train_auc' : history.history['auc'], 'val_auc' : history.history['val_auc'],
            'train_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']}
    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"figures/{name}_training_results.csv", index=False, header=True)

def compute_roc(model, name, test_images, test_labels):
    preds = model.predict(test_images)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, preds)
    df = pd.DataFrame.from_dict({'fpr' : fpr, 'tpr' : tpr, 'threshold' : thresholds})
    df.to_csv(f"figures/{name}_test_roc.csv", index=False, header=True)

def fit_model(data, labels, name, drop=.3):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=42)
    print("Completed splitting")
    model = get_model(width=128, height=128, depth=32, drop=drop)
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"{name}_model.h5", save_best_only=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc", "AUC"],
    )

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    epochs = 50
    history = model.fit(
        X_train, y_train,
        validation_split=.2,
        epochs=epochs,
        shuffle=True,
        batch_size = 32,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    test_loss, test_acc, test_auc = model.evaluate(X_test,  y_test, verbose=2)
    print(f"{name} Test Accuracy: {test_acc}, Test AUC: {test_auc}")
    visualize_performance(history, name)
    compute_roc(model, name, X_test, y_test)


def main():
    metadatafp = 'image_metadata.csv'
    df = pd.read_csv(metadatafp)
    df = df.sort_values(by=['patient'])
    df = df[df['patient'] != 'Breast_MRI_127']
    n_cases = df.shape[0]
    #n_cases = 100
    df = df.head(n_cases)
    data_dir = '../final-project/data'
    # data_dir = '/home/tomasbencomo/final-project/data'
    scans = loader.load_cases(df, data_dir, n_cases, loader.resize_volume)
    er_labels = df['ER'].values
    pr_labels = df['PR'].values
    her2_labels = df['HER2'].values
    print("Completed loading!")

    for i in range(3):
        print(er_labels[i])

    scans = np.stack(scans)
    print(f"Numpy scans shape: {scans.shape}")
    
    scans = scans.reshape((scans.shape[0], 128, 128, 32, 1))
    print(f"Numpy scans shape: {scans.shape}")

    er_labels = er_labels.reshape((er_labels.shape[0], 1))
    print(f"ER Label shape: {er_labels.shape}")

    fit_model(scans, er_labels, "ER_dropout_small_classifier", drop=.3)
    fit_model(scans, her2_labels, "HER2_dropout_small_classifier", drop=.3)
    fit_model(scans, pr_labels, "PR_dropout_small_classifier", drop=.3)
    fit_model(scans, er_labels, "ER_dropout_medium_classifier", drop=.6)
    fit_model(scans, her2_labels, "HER2_dropout_medium_classifier", drop=.6)
    fit_model(scans, pr_labels, "PR_dropout_medium_classifier", drop=.6)
    fit_model(scans, er_labels, "ER_dropout_high_classifier", drop=.9)
    fit_model(scans, her2_labels, "HER2_dropout_high_classifier", drop=.9)
    fit_model(scans, pr_labels, "PR_dropout_high_classifier", drop=.9)

    
if __name__ == '__main__':
    main()
