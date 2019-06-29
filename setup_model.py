import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam
from tflearn import DNN
import time

class Model(object):

    def __init__(
        self,
        optimizer = 'adam',
        optimizer_param = 0.95,
        learning_rate = 0.016,
        learning_rate_decay = 0.864,
        decay_step = 50,
        dropout = 0.956,
        activation = 'relu',
        epochs = 15,
        is_training = True
    ):
        self.optimizer = optimizer,
        self.optimizer_param = optimizer_param,
        self.learning_rate = learning_rate,
        self.learning_rate_decay = learning_rate_decay,
        self.decay_step = decay_step,
        self.dropout = dropout,
        self.activation = activation,
        self.epochs = epochs,
        self.is_training = is_training

    def cnn(self):

        images = input_data(shape=[None, 48, 48, 1], name='input')
        images = conv_2d(images, 64, 3, padding='same', activation=self.activation)
        images = batch_normalization(images)
        #max pooling
        images = max_pool_2d(images, 3, strides=2)
        images = conv_2d(images, 128, 3, padding='same', activation=self.activation)
        images = max_pool_2d(images, 3, strides=2)
        images = conv_2d(images, 256, 3, padding='same', activation=self.activation)
        #maxpooling
        images = max_pool_2d(images, 3, strides=2)
        images = dropout(images, keep_prob=self.dropout)
        images = fully_connected(images, 4096, activation=self.activation)
        images = dropout(images, keep_prob=self.dropout)
        #fully connected layers
        images = fully_connected(images, 1024, activation=self.activation)
        images = fully_connected(images, 7, activation='softmax')

        if self.optimizer == 'momentum':

            optimizers = Momentum(
                learning_rate=self.learning_rate,
                momentum=self.optimizer_param,
                lr_decay=self.learning_rate_decay,
                decay_step=self.decay_step
            )
        elif self.optimizer == 'adam':

            optimizers = Adam(
                learning_rate=self.learning_rate,
                beta1=self.optimizer_param,
                beta2=self.learning_rate_decay,
            )
        else:
            print("Error Optimizer")

        network = regression(
            images,
            optimizer=optimizers,
            loss='categorical_crossentropy',
            learning_rate=self.learning_rate,
            name='output'
        )

        return network

    def train(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val
    ):

        with tf.Graph().as_default():
            print("Building Model...........")
            network = build_CNN()
            model = DNN(
                network,
                tensorboard_dir="path_to_logs",
                tensorboard_verbose=0,
                checkpoint_path="path_to_checkpoints",
                max_checkpoints=1
            )

            if self.is_training:
                # Training phase

                print("start training...")
                print("  - emotions = {}".format(7))
                print("  - optimizer = '{}'".format(self.optimizer))
                print("  - learning_rate = {}".format(0.016))
                print("  - learning_rate_decay = {}".format(self.learning_rate_decay))
                print("  - otimizer_param ({}) = {}".format(self.optimizer, self.optimizer_param))
                print("  - Dropout = {}".format(self.dropout))
                print("  - epochs = {}".format(self.epochs))

            start_time = time.time()
            model.fit(

                {'input': X_train.reshape(-1, 48, 48, 1)},

                {'output': Y_train},

                validation_set=(
                    {'input': X_val.reshape(-1, 48, 48, 1)},

                    {'output': Y_val},
                ),
                batch_size=128,
                n_epoch=10,
                show_metric=True,
                snapshot_step=100

            )

            training_time = time.time() - start_time
            print("training time = {0:.1f} sec".format(training_time))
            print("saving model...")
            model.save("saved_model.bin")