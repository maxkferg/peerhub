from dataset import DataGenerator, TRAIN, VAL

# path to the model weights files.
img_width, img_height = 224, 224
img_channels = 3
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 64


params = {
    'dim': (img_height, img_width, img_channels),
    'batch_size': batch_size,
    'n_classes': 24,
    'shuffle': True,
}


def test_dataset(generator):
    for i in range(10):
        X,y = generator[i]
        _,r = generator.__getitem__(i,raw=True)

        for i in range(len(y)):
            name = "task_%i"%i
            labels = y[name]
            for row in range(labels.shape[0]):
                print("(%i) Name:"%i, name, "Raw label:", r[row,i], "Categorical:", labels[row,:])



def test(train_data_dir):
    # Prepare data augmentation configuration
    train_generator = DataGenerator(train_data_dir, TRAIN, **params)
    val_generator = DataGenerator(train_data_dir, VAL, **params)

    test_dataset(train_generator)
    test_dataset(val_generator)
