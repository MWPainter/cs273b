from tflearn.layers.conv import conv_2d, max_pool_1d, conv_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_TRAINING_SAMPLES = 1235242
NUM_VALIDATION_SAMPLES = 45000
LEARNING_RATE = 0.002
MODEL_NAME = ('hard_multitask-samples-{}-lr-{}'.format(NUM_TRAINING_SAMPLES, LEARNING_RATE))
TRAIN_PATH = '/mnt/logs/train/'
MODEL_PATH = '/mnt/logs/model/'
N_TRAIN_STEPS = 1
NUM_EPOCHS = 10

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
            features = np.load("/mnt/data/train_x.npy")
            labels = np.load("/mnt/data/train_y.npy")
        else:
            features = np.load("/mnt/data/val_x.npy")
            labels = np.load("/mnt/data/val_y.npy")
        indices = np.random.choice(np.arange(len(features)), batch_size, replace=False)
        inputs = features[indices]
        labels = labels[indices]

        return inputs, labels

def get_metrics(predictions, labels, y_conv):
	acc, recall, precision, auc_score, msqe = 0, 0, 0, 0, 0
	for i in range(164):
		acc += accuracy_score(predictions[:,i], labels[:,i])
		recall += recall_score(predictions[:,i], labels[:,i])
		precision += precision_score(predictions[:,i], labels[:,i])
		try:
			auc_score += roc_auc_score(predictions[:,i], labels[:,i])
		except ValueError:
			pass

	msqe = mean_squared_error(labels, y_conv)
	auc_score  /= 164
	acc  /= 164
	precision  /= 164
	recall /= 164
	return acc, recall, precision, auc_score, msqe

def train_conv_net(model, model_name):
    for i in range(N_TRAIN_STEPS):
        
        #x_train, y_train = get_batch(int(512), True) 
        #x_val_set, y_val_set = get_batch(int(124), False)            
        x_train = np.load("/mnt/data/train_x.npy")
        y_train = np.load("/mnt/data/train_y.npy")
        x_val_set = np.load("/mnt/data/val_x.npy")
        y_val_set = np.load("/mnt/data/val_y.npy")

        model.fit(x_train, y_train, n_epoch=NUM_EPOCHS, batch_size=256, validation_set=(x_val_set, y_val_set), show_metric=True, shuffle=True)
        print(metrics)
        predictions = model.predict(x_val_set)
        y_conv = predictions
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        acc, recall, precision, auc_score, msqe = get_metrics(predictions, y_val_set, y_conv)
        print(acc) 
        print(msqe)
        print(recall)
        print(precision)
        print(auc_score)     

graph = tf.Graph()
with graph.as_default():
    train_model = model()
    train_conv_net(train_model, MODEL_NAME)
