from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer
from tensorflow.keras.models import Sequential

from draw_convnet import plot_keras_convnet

model = Sequential([
    InputLayer(input_shape=(28, 28, 1)),
    Conv2D(filters=6, kernel_size=5, padding="valid", activation="relu"),
    MaxPool2D(pool_size=2),
    Conv2D(filters=16, kernel_size=5, padding="valid", activation="relu"),
    MaxPool2D(pool_size=2),
    Flatten(),
    Dense(units=120, activation="relu"),
    Dense(units=84, activation="relu"),
    Dense(units=10, activation="softmax"),
])
plot_keras_convnet(model, font_size=6, to_file="assets/convnet_fig.svg")
