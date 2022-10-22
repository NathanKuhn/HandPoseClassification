import pandas as pd
import numpy as np
import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(63),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(14)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # used with a true index for the output

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model

def main():
    # Get data from csv files
    data_0 = pd.read_csv("training_dataset/label_0.csv")
    data_1 = pd.read_csv("training_dataset/label_1.csv")
    data_2 = pd.read_csv("training_dataset/label_2.csv")
    data_3 = pd.read_csv("training_dataset/label_3.csv")
    data_4 = pd.read_csv("training_dataset/label_4.csv")
    data_5 = pd.read_csv("training_dataset/label_5.csv")
    data_6 = pd.read_csv("training_dataset/label_6.csv")
    data_7 = pd.read_csv("training_dataset/label_7.csv")
    data_8 = pd.read_csv("training_dataset/label_8.csv")
    data_9 = pd.read_csv("training_dataset/label_9.csv")
    data_10 = pd.read_csv("training_dataset/label_10.csv")
    data_11 = pd.read_csv("training_dataset/label_11.csv")
    data_12 = pd.read_csv("training_dataset/label_12.csv")
    data_13 = pd.read_csv("training_dataset/label_13.csv")

    train_data = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13])
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