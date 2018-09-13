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
    for i in range(100):
        X,y = generator[i]
        _,r = generator.__getitem__(i,raw=True)
        print("Image Batch Shape:",X.shape)
        print("Label batch shape:",y.shape)
        print("Labels: ",y[i,:],"\n")
        print("Raw labels:", r[i,:],"\n")

        for t,end in enumerate(generator.tasks):
            start = 0 if t==0 else generator.tasks[t-1]
            true = y[i, start:end]
            print("Task %i"%t, "target:", true)



def test(train_data_dir):
    # Prepare data augmentation configuration
    train_generator = DataGenerator(train_data_dir, TRAIN, **params)
    val_generator = DataGenerator(train_data_dir, VAL, **params)

    test_dataset(train_generator)