import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Define the input feature: total rooms.
input_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the target
targets = california_housing_dataframe["median_house_value"]

# Configure LinearRegressor
# Train with GradientDescentOptimizer
# which implements Mini-Batch Stochastic Gradient Descent (SGD)
# Learning rate controls size of gradient step

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=optimizer)


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


linear_regressor.train(
    input_fn=lambda: input_fn(input_feature, targets),
    steps=100)
