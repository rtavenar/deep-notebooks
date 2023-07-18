from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense

from draw_convnet import plot_keras_convnet

model = Sequential([
    InputLayer(input_shape=(32, 32, 3)),
    Conv2D(filters=16, kernel_size=3, padding="same"),
    MaxPool2D(pool_size=2),
    Conv2D(filters=32, kernel_size=5, padding="valid"),
    AvgPool2D(pool_size=2),
    Flatten(),
    Dense(units=64),
    Dense(units=10)
])

plot_keras_convnet(model, font_size=6, to_file="assets/model_cnn.svg")