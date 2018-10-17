import os
import math
import keras
import imagehash
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from utils import invert_map

TRAIN = "train"
VAL = "val"


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, directory, mode=TRAIN, batch_size=32, dim=(224,224,3), n_classes=10, shuffle=True):
        """Initialization"""
        print("Loading dataset from %s"%directory)
        self.dim = dim
        self.directory = directory
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = self.load_data(os.path.join(directory, 'X_train.npy'))
        self.labels = self.load_labels(os.path.join(directory, 'y_train.npy'))
        self.list_IDs = self.__get_list_IDs(mode)
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index, raw=False):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, raw)

        return X, y


    def load_data(self,filename):
        """Load the data from a file"""
        return np.load(filename)


    def load_labels(self,filename):
        """Load the data from a file"""
        self.raw_labels = np.load(filename)
        return self.__to_onehot(self.raw_labels)


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __get_list_IDs(self, mode):
        """Return the IDs that make up this set"""
        ids = range(self.data.shape[0])
        if mode == TRAIN:
            ids = [i for i in ids if (i%5)>0]
        elif mode == VAL:
            ids = [i for i in ids if (i%5)==0]
        else:
            raise ValueError("Unknown mode %s"%mode)
        print("Filtered %s dataset contains %i samples"%(mode, len(ids)))
        return ids


    def __to_onehot(self, labels):
        """
        Convert a block of data to onehot labels
        Return the task indices and the one hot labels
        [2,1,0] -> {task0: [[0,0,1]], task2: [[0,1]] ...}
        """
        new_labels = {}
        
        # labels should be a column vector (consistency between combined/single)
        if len(labels.shape)==1:
            return {"task_0": np_utils.to_categorical(labels)}

        for t in range(labels.shape[1]):
            name = "task_%i"%t
            onehot = np_utils.to_categorical(labels[:,t])
            # Fix up the labels that should be set to -1
            for row in range(onehot.shape[0]):
                if labels[row,t] < 0:
                    onehot[row,:] = -1
            # Now the labels for this task can be recorded
            new_labels[name] = onehot
            num_classes.append(onehot.shape[1])
            # Print a test for debugging
            print("Name",name, "Raw label:", labels[0,t], "One hot:", new_labels[name][0,:])
        return new_labels


    def __data_generation(self, list_IDs_temp, raw):
        """Generates data containing batch_size samples"""
        X = self.data[list_IDs_temp,:,:,:]
        if raw:
            y = self.raw_labels[list_IDs_temp]
        else:
            y = {}
            for k,v in self.labels.items():
                y[k] = v[list_IDs_temp,:]
        return X, y


def show_images(img1, img2):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(img2)
    plt.show()



def get_image_hash(image):
    image = Image.fromarray(image, 'RGB')
    return str(imagehash.dhash(image))



def build_image_hashmap(directories):
    """
    Build a map between hashes and images
    """
    hashmap = defaultdict(list)
    for d in directories:
        data = np.load(os.path.join(d, 'X_train.npy'))
        for i in range(data.shape[0]):
            image = data[i,:,:,:]
            h = get_image_hash(image)
            hashmap[h].append(image)
            #if len(hashmap[h])==2:
            #   show_images(hashmap[h][0], hashmap[h][1])
        print("Found %i unique hashes"%len(hashmap))
    return hashmap



def build_dataset(directories, output_dir):
    print("Building dataset with:", directories)
    print("Finding unique images...")
    hashes = build_image_hashmap(directories)

    # Build a map in the form {hash: index}
    lookup = dict((h,i) for (i,h) in enumerate(hashes.keys()))

    # The new dataset
    n_tasks = len(directories)
    n_hashes = len(hashes)
    X = np.zeros((n_hashes,224,224,3), dtype=np.uint8)
    Y = -1*np.ones((n_hashes,n_tasks), dtype=np.uint8)

    for t, task_dir in enumerate(directories):
        print("Processing task %i"%t)
        data = np.load(os.path.join(task_dir, 'X_train.npy'))
        labels = np.load(os.path.join(task_dir, 'y_train.npy'))

        # Iterate over images in this task
        for i in range(data.shape[0]):
            image = data[i,:,:,:]
            h = get_image_hash(image)
            if h in lookup:
                index = lookup[h]
                X[index] = image
                Y[index, t] = labels[i]
            else:
                print("Hash miss",h)

    X_path = os.path.join(output_dir, 'X_train.npy')
    Y_path = os.path.join(output_dir, 'y_train.npy')
    print("Saving")
    np.save(X_path, X)
    np.save(Y_path, Y)
    print("Saved new dataset to %s"%output_dir)



