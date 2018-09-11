'''
Train a keras model for PHI

Usage:
python model.py --dataset /Volumes/Flash/Data/PHI/t1

'''

import sys
import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from dataset import DataGenerator, TRAIN, VAL

# path to the model weights files.
top_model_weights_path = 'weights/multinet.h5'
img_width, img_height = 224, 224
img_channels = 3
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


params = {
    'dim': (img_height, img_width, img_channels),
    'batch_size': 64,
    'n_classes': 6,
    'shuffle': True,
}

flags = tf.app.flags
flags.DEFINE_string(name='dataset', default='/Volumes/Flash/Data/PHI/t1', help='path to dataset')


def train(train_data_dir):
    model = applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(img_height, img_width, img_channels))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(1, activation='sigmoid'))

    features = Flatten(input_shape=model.output_shape[1:])(model.output)
    features = Dense(256, activation='relu')(features)
    features = Dropout(0.5)(features)
    output = Dense(1, activation='sigmoid')(features)

    model = Model(inputs=model.input, outputs=output)
    print(model.summary())

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    # top_model.load_weights(top_model_weights_path)


    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:-3]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_generator = DataGenerator(train_data_dir, TRAIN, **params)
    val_generator = DataGenerator(train_data_dir, VAL, **params)

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=val_generator,
        nb_val_samples=nb_validation_samples)


def main(args):
    train(flags.FLAGS.dataset)


if __name__ == '__main__':
    tf.app.run()

