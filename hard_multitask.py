from tflearn.layers.conv import conv_2d, max_pool_1d, conv_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn
import numpy as np

NUM_TRAINING_SAMPLES = 1000
LEARNING_RATE = 0.002
MODEL_NAME = ('hard_multitask-samples-{}-lr-{}'.format(NUM_TRAINING_SAMPLES, LEARNING_RATE))
TRAIN_PATH = 'logs/train/'
MODEL_PATH = 'logs/model'
N_TRAIN_STEPS = 10
NUM_EPOCHS = 200

def model(lr = LEARNING_RATE):
    network = input_data(shape=[None, 600, 4], name='features')
    network = conv_1d(network, 300, 19, strides=1, activation='relu')
    network = max_pool_1d(network, 3, strides=3)
    network = conv_1d(network, 200, 11, strides=1, activation='relu')
    network = max_pool_1d(network, 4, strides=4)
    network = conv_1d(network, 200, 7, strides=1, activation='relu')
    network = max_pool_1d(network, 4, strides=4)
    network = fully_connected(network, 1000, activation='relu')
    network = fully_connected(network, 1000, activation='relu')
    network = fully_connected(network, 164, activation='sigmoid')
    network = regression(network, optimizer='rmsprop', loss='binary_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model

def get_batch(batch_size, is_train):
        inputs, labels = [], []
        if is_train:
            features = np.load("train_x.npy")
            labels = np.load("train_y.npy")
        else:
            features = np.load("val_x.npy")
            labels = np.load("val_y.npy")
        indices = np.random.choice(np.arange(len(features)), batch_size, replace=False)
        inputs = features[indices]
        labels = labels[indices]

        return [inputs, labels]

def train_conv_net(model, model_name, num_epoch):
    for i in range(N_TRAIN_STEPS):
        
        features_train, labels_train = get_batch(int(NUM_TRAINING_SAMPLES * 0.8), True) 
        features_val_set, labels_val_set = get_batch(int(NUM_TRAINING_SAMPLES * 0.1), False)            
        print(model.fit({'features': features_train}, {'labels': labels_train}, n_epoch=NUM_EPOCHS, 
                validation_set=({'features': features_val_set}, {'labels': labels_val_set}), 
                shuffle=True, snapshot_step=None, show_metric=True, 
                run_id=model_name))      

graph = tf.Graph()
with graph.as_default():
    train_model = model()
    train_conv_net(train_model, MODEL_NAME, 100)