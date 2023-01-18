import pandas as pd
import numpy as np
import tensorflow as tf
import os

DATASET_PATH = "training_dataset/"

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(63),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(9)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # used with a true index for the output

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model

def main():
    # Get data from csv files
    files = os.listdir(DATASET_PATH)
    train_data = pd.concat([pd.read_csv(DATASET_PATH + data_file) for data_file in files])
    train_data.pop("Unnamed: 0") # left over index from writing to csv in recordData.py

    # Shuffle data
    train_data = train_data.sample(frac=1)

    # Get labels
    label_data = train_data.pop("63")

    # Add gaussian noise
    data_array = np.array(train_data)
    data_array += np.random.normal(0, 0.07, data_array.shape)
    train_data = pd.DataFrame(data_array)

    # Create and train model
    model = create_model()
    model.fit(train_data, label_data, epochs=5, validation_split=0.05)
    model.save_weights('weights/checkpoint')

if __name__ == "__main__":
    main()