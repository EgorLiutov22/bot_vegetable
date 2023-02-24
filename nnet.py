import tensorflow as tf
# import tensorflow_hub as hub
# from keras.applications.resnet50 import decode_predictions
import numpy as np
import cv2


def preprocess_image(image):
    image = np.array(image)

    sample_image_resized = cv2.resize(image, (224, 224))

    sample_image = np.expand_dims(sample_image_resized, axis=0)

    return sample_image


class Network:
    def __init__(self, image):
        self.__model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        self.__image = preprocess_image(image)

    def prediction(self):
        predictions = self.__model.predict(self.__image)
        p = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]
        return p[0][1]
