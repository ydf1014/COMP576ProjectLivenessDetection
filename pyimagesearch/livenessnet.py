import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.pooling import AveragePooling2D
from keras import backend as K
from keras.applications import VGG16, InceptionV3, ResNet50


class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = tf.keras.Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model


class Liveness_VGG16:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = VGG16(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_InceptionV3:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = InceptionV3(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet50:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape=input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

