import os
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, MaxPool2D, ReLU, Input, ZeroPadding2D, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from PIL import Image
import os

N_CLASSES = 6

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def double_conv2d(output_channels):
    return Sequential([
        Conv2D(filters=output_channels, kernel_size=3, padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=output_channels, kernel_size=3, padding="same"),
        BatchNormalization(),
        ReLU()
    ])

def down_sampling_block(output_channels):
    return Sequential([
        MaxPool2D(pool_size=2),
        double_conv2d(output_channels)
    ])

def up_sampling_block(output_channels, shape1, shape2):
    x1 = Input(shape=shape1)
    x2 = Input(shape=shape2)
    x1_up = Conv2DTranspose(output_channels, kernel_size=2, strides=2)(x1)
    
    # input is HWC
    diffY = x2.get_shape().as_list()[1] - x1_up.get_shape().as_list()[1]
    diffX = x2.get_shape().as_list()[2] - x1_up.get_shape().as_list()[2]

    x1_up = ZeroPadding2D(padding=((diffY // 2, diffY - diffY // 2), (diffX // 2, diffX - diffX // 2)))(x1_up)
    x = tf.concat([x2, x1_up], axis=-1)
    return Model(inputs=[x1, x2], outputs=double_conv2d(output_channels)(x))

def unet_model(image_size, n_classes):
    inputs = Input(shape=image_size + (3,))
    x0 = double_conv2d(64)(inputs)
    
    x1 = down_sampling_block(128)(x0)
    x2 = down_sampling_block(256)(x1)
    x3 = down_sampling_block(512)(x2)
    x4 = down_sampling_block(1024)(x3)
    
    x = up_sampling_block(512, shape1=x4.shape[1:], shape2=x3.shape[1:])((x4, x3))
    x = up_sampling_block(256, shape1=x.shape[1:], shape2=x2.shape[1:])((x, x2))
    x = up_sampling_block(128, shape1=x.shape[1:], shape2=x1.shape[1:])((x, x1))
    x = up_sampling_block(64, shape1=x.shape[1:], shape2=x0.shape[1:])((x, x0))
    
    outputs = Conv2D(filters=n_classes, kernel_size=1, activation="softmax")(x)
    
    return Model(inputs, outputs)

class VaihingenDataset(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, folder, small_subset=False):
        self.batch_size = batch_size
        self.input_img_paths = sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".png")
        ])
        if small_subset:
            self.input_img_paths = self.input_img_paths[:2 * batch_size]
        self.normalization_means = np.array([0.4643, 0.3185, 0.3141])
        self.normalization_stds = np.array([0.2171, 0.1561, 0.1496])

    def __len__(self):
        return max(1, len(self.input_img_paths) // self.batch_size)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = []
        y = []
        for j, path in enumerate(batch_input_img_paths):
            img = Image.open(path).convert("RGB")
            x.append(np.asarray(img) / 255.)
            y.append(np.load(path.replace(".png", "_gt.npy")))
        x = np.array(x)
        y = to_categorical(np.array(y), num_classes=N_CLASSES)
        x -= self.normalization_means[None, None, None, :]
        x /= self.normalization_stds[None, None, None, :]
        return x, y

train_loader = VaihingenDataset(batch_size=4, folder="vaihingen-cropped/vaihingen_train/")
val_loader = VaihingenDataset(batch_size=4, folder="vaihingen-cropped/vaihingen_test/")


model = unet_model(image_size=(512, 512), n_classes=N_CLASSES)
model.compile(optimizer=Adam(learning_rate=1e-3), loss=categorical_focal_loss(alpha=[[0.25] * N_CLASSES], gamma=2), metrics=["accuracy"])
cb = ModelCheckpoint("unet_{epoch:02d}.h5", save_best_only=True)
model.fit(train_loader, epochs=20, validation_data=val_loader, callbacks=[cb])
