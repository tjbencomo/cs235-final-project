import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
import loader

def get_model(width, height, depth):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def main():
    metadatafp = 'image_metadata.csv'
    df = pd.read_csv(metadatafp)
    df = df.sort_values(by=['patient'])
    df = df[df['patient'] != 'Breast_MRI_127']
    n_cases = df.shape[0]
    df = df.head(n_cases)
    data_dir = '../final-project/data'
    # data_dir = '/home/tomasbencomo/final-project/data'
    scans = loader.load_cases(df, data_dir, n_cases, loader.resize_volume)
    er_labels = df['ER'].values
    pr_labels = df['PR'].values
    her2_labels = df['HER2'].values
    print("Completed loading!")

    scans = np.stack(scans)
    print(f"Numpy scans shape: {scans.shape}")
    
    scans = scans.reshape((scans.shape[0], 128, 128, 64, 1))
    print(f"Numpy scans shape: {scans.shape}")

    er_labels = er_labels.reshape((er_labels.shape[0], 1))
    print(f"ER Label shape: {er_labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(scans, er_labels, test_size=.2, random_state=42)
    print("Completed splitting")

    # Build model.
    model = get_model(width=128, height=128, depth=64)
    #model.summary()

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    epochs = 20
    model.fit(
        X_train, y_train,
        validation_split=.1,
        epochs=epochs,
        shuffle=True,
        batch_size = 8,
        callbacks=[early_stopping_cb]
    )


if __name__ == '__main__':
    main()
