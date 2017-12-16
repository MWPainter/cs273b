import numpy as np
import tensorflow as tf
from softcnn import softcnn






# Load in the model
run_name = "soft_small_set_0"
data_path = "/mnt/data/"
basedir = "/mnt/" + run_name + "/"
logdir = basedir + "logs/"
model = softcnn(run_name, data_path, batch_size = 50, num_epochs = 100, weight_decay = 0.01, basedir = basedir, logdir = logdir)
model.make_graph()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/mnt/soft_small_set_0/checkpoints/soft_small_set_0_epoch_15_step_13500.ckpt")
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print v



# Load in the evaluation set


# Get predicition scores



# 
