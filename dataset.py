import os
import keras
import numpy as np

TRAIN = "train"
VAL = "val"



class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, directory, mode=TRAIN, batch_size=32, dim=(224,224,3), n_classes=10, shuffle=True):
        """Initialization"""
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


    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def load_data(self,filename):
    	"""Load the data from a file"""
    	return np.load(filename)


    def load_labels(self,filename):
    	"""Load the data from a file"""
    	return np.load(filename)


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


    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        X = self.data[list_IDs_temp,:,:,:]
        y = self.labels[list_IDs_temp]

        return X, y


