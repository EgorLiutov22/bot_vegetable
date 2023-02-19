import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np
import cv2


class Network:
    def __init__(self, image):
        # self.__model = tf.keras.Sequential([
        #     hub.KerasLayer("https://tfhub.dev/sayakpaul/distill_bit_r50x1_160_feature_extraction/1", trainable=True)
        # ])
        self.__model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        self.__image = self.preprocess_image(image)

    def preprocess_image(self, image):
        image = np.array(image)
        # # Resize to (160, 160)
        # image_resized = tf.image.resize(image, (160, 160))
        # img_reshaped = tf.reshape(image_resized, [1, image.shape[0], image.shape[1], image.shape[2]])
        # # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        # image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
        # sample_image = cv2.imread(str(image))
        #
        sample_image_resized = cv2.resize(image, (224, 224))

        sample_image = np.expand_dims(sample_image_resized, axis=0)

        return sample_image

    def prediction(self):
        predictions = self.__model.predict(self.__image)
        return decode_predictions(predictions, top=3)[0]
