'''
Train a keras model for PHI

Usage:
python model.py --dataset /Volumes/Flash/Data/PHI/t1


python model.py --prepare /Volumes/Flash/Data/PHI/

python model.py --dataset /Volumes/Flash/Data/PHI/ --test=True
'''

import sys, os
import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from dataset import DataGenerator, TRAIN, VAL
from losses import crossentropy_filtered_loss
from metrics import accuracy_fn
from utils import list_directories
from dataset import build_dataset
from test import test


# path to the model weights files.
top_model_weights_path = 'weights/multinet.h5'
img_width, img_height = 224, 224
img_channels = 3
nb_train_steps = 1000
nb_validation_steps = 20
epochs = 50
batch_size = 64


params = {
    'dim': (img_height, img_width, img_channels),
    'batch_size': batch_size,
    'n_classes': 24,
    'shuffle': True,
}


flags = tf.app.flags
flags.DEFINE_string(name='dataset', default='/Volumes/Flash/Data/PHI/t1', help='path to dataset')
flags.DEFINE_string(name='prepare', default='', help='prepare the dataset using this directory')
flags.DEFINE_string(name='model', default='resnet', help='Feature extraction model')
flags.DEFINE_bool(name='test', default=False, help='run sanity checks')



def train(train_data_dir, model_fn):
    base_model = model_fn(include_top=False, weights='imagenet')
    print('Model loaded:', model_fn)

    # build a classifier model to put on top of the convolutional model
    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    # top_model.load_weights(top_model_weights_path)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    #for layer in model.layers[:-4]:
    #    layer.trainable = False

    # Prepare data augmentation configuration
    train_generator = DataGenerator(train_data_dir, TRAIN, **params)
    val_generator = DataGenerator(train_data_dir, TRAIN, **params)
    _,y = train_generator[0]

    #features = Flatten(input_shape=model.output_shape[1:])(model.output)
    #features = Dense(128, activation='relu')(features)

    outputs = []
    tasks = sorted(y.keys())[:1]
    print("Creating network heads for ", tasks)

    for task in tasks:
        n = y[task].shape[1]
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(n, activation='softmax', name=task)(x)
        #features = Dense(64, activation='relu')(features)
        #features = Dropout(0.2)(features)
        #num_classes = y[task].shape[1]
        #logits = Dense(num_classes, activation='softmax', name=task)(features)
        outputs.append(predictions)

    # Make the multi-head model
    model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',#_filtered_loss,
                  optimizer='rmsprop',
                  metrics=['accuracy',accuracy_fn,])

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_steps,
        epochs=2,
        validation_data=val_generator,
        validation_steps=nb_validation_steps)

    #for layer in model.layers[:249]:
    #    layer.trainable = False
    #for layer in model.layers[249:]:
    #    layer.trainable = True
    
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                  loss='categorical_crossentropy',#crossentropy_filtered_loss,
                  metrics=['accuracy',accuracy_fn])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_steps,
        epochs=50,
        validation_data=val_generator,
        validation_steps=nb_validation_steps)


def get_model_fn(name):
    #return applications.inception_v3.InceptionV3
    if name=="resnet":
        return applications.resnet50.ResNet50
    elif name=="densenet":
        return applications.densenet.DenseNet121
    elif name=="xception":
        return applications.xception.Xception
    elif name=="inception_resnet":
        return applications.inception_resnet_v2.InceptionResNetV2
    else:
        raise ValueError("Unknown model %s"%name)


def main(args):
    if flags.FLAGS.prepare:
        directories = list_directories(flags.FLAGS.prepare)
        dataset = os.path.join(flags.FLAGS.prepare, 'combined')
        if not os.path.exists(dataset):
            os.makedirs(dataset)
        build_dataset(directories, dataset)
    elif flags.FLAGS.test:
        test(flags.FLAGS.dataset)
    else:
        model_fn = get_model_fn(flags.FLAGS.model)
        train(flags.FLAGS.dataset, model_fn)


if __name__ == '__main__':
    tf.app.run()

