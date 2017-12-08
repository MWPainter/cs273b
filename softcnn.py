import tensorflow as tf
import numpy as np
import numpy.random as npr
import layers
import utils
import seaborn as sns
import matplotlib
import time
from tqdm import trange
import os
matplotlib.use("Agg")
import matplotlib as plt


def mean(arr):
    return np.mean(np.array(arr))


class softcnn():

    '''
    initialiations
    '''
    def __init__(self, run_name, data_path, batch_size = 50, num_epochs = 100,  input_shape = [600,4], num_labels = 164,
                 basedir = "", logdir = "", learning_rate = 0.002, weight_decay = 0.01, if_save_model = True, save_frequency = 5, test_eval_frequency = 50, val_eval_frequency = 50, 
                 val_batch_size = 50):

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.curr_step  = 0
        self.curr_epoch = 0
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.run_name = run_name
        self.if_save_model = if_save_model
        self.save_frequency = save_frequency
        self.test_eval_frequency = test_eval_frequency
        self.val_eval_frequency = val_eval_frequency
        self.val_batch_size = val_batch_size
        self.basedir = basedir
        self.logdir = logdir

        # load in data - convert labels from (None, 164) to (None, 164, 2)
        # for 0 labels, pick 0th column from Identity matrix, and 1 labels pick 1st column
        self.train_x = np.load(data_path + "train_x.npy")
        self.train_y = np.load(data_path + "train_y.npy")
        self.train_y = np.eye(2)[self.train_y]      
        self.val_x = np.load(data_path + "val_x.npy")
        self.val_y = np.load(data_path + "val_y.npy")
        self.val_y = np.eye(2)[self.val_y]      
        self.test_x = np.load(data_path + "test_x.npy")
        self.test_y = np.load(data_path + "test_y.npy")
        self.test_y = np.eye(2)[self.test_y]      
        self.cell_types = np.load(data_path + "target_labels.npy")
        self.class_ratios = self.compute_train_set_class_imballances(np.load(data_path + "train_y.npy"))
        self.example_weightings = self.train_y[:,:,0] * self.class_ratios + self.train_y[:,:,1] * (1 - self.class_ratios)

        self.iterations = (self.train_x.shape[0] + batch_size - 1) / batch_size

        # FOR TESTING WITH OVERFITTING:
        self.train_x = self.train_x[:5000]
        self.train_y = self.train_y[:5000,:41]#3]
        self.val_x = self.val_x[:500]
        self.val_y = self.val_y[:500,:41]#3]
        self.num_labels = 41#3
        self.example_weightings = self.example_weightings[:5000,:41]#3]

        self.iterations = (self.train_x.shape[0] + batch_size - 1) / batch_size

        # Seed npr, can change this to make det behaviour
        npr.seed(int(time.time()))

        #sets up the new tf session
        tf.reset_default_graph()        
        self.sess = tf.Session()
        
        #sets up the graph of the network
        with tf.variable_scope("soft_cnn"):
            self.make_graph()

        #sets up the writer for tensorboard functionality
        if not os.path.isdir(basedir):
            os.mkdir(basedir)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        if not os.path.isdir(basedir+"/result_graphs"):
            os.mkdir(basedir+"/result_graphs")

        if not os.path.isdir(basedir+"/checkpoints"):
            os.mkdir(basedir+"/checkpoints")

        self.writer = tf.summary.FileWriter(logdir, self.sess.graph)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()






    def make_graph(self):
        """
        Constructs the whole graph, consisting of all of the (tied) models
        Initialize total_loss to 0, but it will accumulate all of the loss functions from the models for each task
        """
        print("-----CONSTRUCTING GRAPH-----")
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1]])
        self.lbls = tf.placeholder(tf.float32, shape=[None, self.num_labels, 2])
        self.weightings = tf.placeholder(tf.float32, shape=[None, self.num_labels])
        self.is_train = tf.placeholder(tf.bool)
        self.losses = []
        self.accuracies, self.aucs, self.msqes, self.recalls, self.precisions = [],[],[],[],[]
        self.accuracy_update_ops, self.auc_update_ops, self.msqe_update_ops, self.recall_update_ops, self.precision_update_ops = [],[],[],[],[]
        self.total_loss = 0

        for i in range(self.num_labels):
            print("----Cell Type: " + self.cell_types[i])
            with tf.variable_scope(self.cell_types[i].replace('+','')):
                lbl = self.lbls[:,i,:]
                lbl = tf.reshape(lbl,[tf.shape(self.lbls)[0], 2])
                self.add_task_model_to_graph(lbl)

        
        print("----Constructing evaluation, loss, optimizaer and tensorboard ops")
        self.update_eval_metrics_ops = [self.accuracy_update_ops, self.auc_update_ops, self.msqe_update_ops, self.recall_update_ops, self.precision_update_ops]
        self.accuracy = tf.divide(tf.add_n(self.accuracies), self.num_labels)
        self.auc = tf.divide(tf.add_n(self.aucs), self.num_labels)
        self.msqe = tf.divide(tf.add_n(self.msqes), self.num_labels)
        self.recall = tf.divide(tf.add_n(self.recalls), self.num_labels)
        self.precision = tf.divide(tf.add_n(self.precisions), self.num_labels)

        self.weight_distance = self.construct_weight_distance()
        self.total_loss /= self.num_labels
        self.objective_loss = self.total_loss
        self.total_loss += self.distance_penalty * self.weight_distance

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        #TENSORBOARD operators
        self.train_loss_op = tf.summary.scalar("train_loss", self.total_loss)
        self.test_loss_op  = tf.summary.scalar("test_loss",  self.total_loss)
        self.train_acc_op = tf.summary.scalar("train_accuracy", self.accuracy)
        self.test_acc_op = tf.summary.scalar("test_accuracy", self.accuracy)
        self.train_prec_op = tf.summary.scalar("train_precision", self.precision)
        self.test_prec_op = tf.summary.scalar("test_precision", self.precision)
        self.train_recall_op = tf.summary.scalar("train_recall", self.recall)
        self.test_recall_op = tf.summary.scalar("test_recall", self.recall)
        self.train_auc_op = tf.summary.scalar("train_auc", self.auc)
        self.test_auc_op = tf.summary.scalar("test_auc", self.auc)

        self.tb_objective_loss_op = tf.summary.scalar("objective_loss", self.objective_loss)
        self.tb_weight_distance_loss_op = tf.summary.scalar("weight_distance_loss", self.distance_penalty * self.weight_distance)
        
        
        print("---------------------------")





    def compute_train_set_class_imballances(self, labels):
        """
        Computes a vector of shape (self.num_labels) with the class imballances present in the dataset with labels 'labels'
        NOTE: this assumes that train_y is currently of shape (num_examples, num_tasks)
        """
        data_set_size = labels.shape[0]
        class_sums = np.sum(labels, axis=0)
        class_ratios = class_sums / float(data_set_size)
        return class_ratios





    def add_task_model_to_graph(self, lbls, tasknum):
        conv1 =  layers.max_pool1d(
                    layers.batch_norm(
                        layers.conv1dLayer(self.x, filterSize = 19, outputDim = 300, stride = 1, name = "conv1"),
                    self.is_train) , size = 3, stride = 3, name = "pool1")

        conv2 =  layers.max_pool1d(
                    layers.batch_norm(
                        layers.conv1dLayer(conv1, filterSize = 11, outputDim = 200, stride =1, name = "conv2"),
                    self.is_train), size = 4, stride = 4, name = "pool2")

        conv3 =  layers.max_pool1d(
                    layers.batch_norm(
                        layers.conv1dLayer(conv2, filterSize = 7, outputDim = 200, stride = 1, name = "conv3"),
                    self.is_train), size = 4, stride = 4, name = "pool3")

        fc1   = layers.batch_norm(layers.denseLayer(conv3, outputDim = 1000, name = "fc1"), self.is_train)
        fc2   = layers.batch_norm(layers.denseLayer(fc1, outputDim = 1000, name = "fc2"), self.is_train)

        y_conv = layers.readoutLayer(fc2, outputDim = 2, name = "readout")
        train_loss = tf.reduce_mean(tf.multiply(self.weightings[:,tasknum], tf.nn.softmax_cross_entropy_with_logits(labels = lbls, logits = y_conv)))
        test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = lbls, logits = y_conv))
        # train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        correct_labels = tf.argmax(lbls,1) # convert back from one hot
        predictions = tf.argmax(y_conv, 1)

        accuracy, accuracy_op   = tf.metrics.accuracy(correct_labels, predictions)
        recall, recall_op       = tf.metrics.recall(correct_labels, predictions)
        precision, precision_op = tf.metrics.precision(correct_labels, predictions)
        auc, auc_op             = tf.metrics.auc(correct_labels, predictions)
        msqe, msqe_op           = tf.metrics.mean_squared_error(correct_labels, predictions)

        self.losses.append(loss)
        self.total_loss += loss

        self.accuracies.append(accuracy)
        self.accuracy_update_ops.append(accuracy_op)
        self.recalls.append(recall)
        self.recall_update_ops.append(recall_op)
        self.precisions.append(precision)
        self.precision_update_ops.append(precision_op)
        self.aucs.append(auc)
        self.auc_update_ops.append(auc_op)
        self.msqes.append(msqe)
        self.msqe_update_ops.append(msqe_op)


        


    def construct_weight_distance(self):
        """
        Constructs the sum of the distances between each pair of corresponding weights
        """
        conv1_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv1_weights:0')]
        conv2_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv2_weights:0')]
        conv3_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv3_weights:0')]

        total_dist = 0
        total_dist += self.pairwise_distances(conv1_weights)
        total_dist += self.pairwise_distances(conv2_weights)
        total_dist += self.pairwise_distances(conv3_weights)





    def pairwise_distances(self, weights):
        """
        Assumes that we are given a set of weights with identical dimensions
        Computes the sum pairwise distances between those weights
        """
        dist = 0
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                dist += tf.nn.l2_loss(weights[i] - weights[j])
        return dist






    def run_iter(self, iter_number):
        """
        Runs a training step + computes loss, given a minibatch
        """
        iter_input, iter_lbls, weights = self.get_batch(iter_number)
        feed_dict = {self.x: iter_input, self.lbls: iter_lbls, self.example_weightings: weights, self.is_train: True}
        self.sess.run(self.train_step, feed_dict=feed_dict)
        self.curr_step += 1





    def evaluate(self):
        """
        Computes evaluation metrics on the val set.  + prints summaries for tensorboard
        """
        batch_input, batch_lbls = self.sample_val_batch()
        acc, loss, recall, precision, auc, msqe = self.eval_batch(batch_input, batch_lbls, False)

        feed_dict = {self.x: batch_input, self.lbls: batch_lbls, self.is_train: False}
        loss_str = self.sess.run(self.test_loss_op, feed_dict=feed_dict)
        acc_str = self.sess.run(self.test_acc_op , feed_dict=feed_dict)
        prec_str = self.sess.run(self.test_prec_op, feed_dict=feed_dict)
        recall_str = self.sess.run(self.test_recall_op, feed_dict=feed_dict)
        auc_str = self.sess.run(self.test_auc_op, feed_dict=feed_dict)
        
        self.writer.add_summary(acc_str, self.curr_step)
        self.writer.add_summary(loss_str, self.curr_step)
        self.writer.add_summary(prec_str, self.curr_step)
        self.writer.add_summary(recall_str, self.curr_step)
        self.writer.add_summary(auc_str, self.curr_step)

        return acc, loss, recall, precision, auc, msqe



    

    def eval_batch(self, iter_input, iter_lbls, is_train):
        """
        Runs the evaluation operators, given a minibatch
        """
        feed_dict = {self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train}
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.update_eval_metrics_ops, feed_dict=feed_dict)
        acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
        loss = self.sess.run(self.total_loss, feed_dict=feed_dict)
        recall = self.sess.run(self.recall, feed_dict=feed_dict)
        precision = self.sess.run(self.precision, feed_dict=feed_dict)
        auc = self.sess.run(self.auc, feed_dict=feed_dict)
        msqe = self.sess.run(self.msqe, feed_dict=feed_dict)
        
        return acc, loss, recall, precision, auc, msqe







    def get_batch(self, iter_num):
        """
        Gets a (mini) batch from the dataset. 
        """
        beg = self.batch_size * iter_num
        end = self.batch_size * (iter_num + 1)
        if end > self.train_x.shape[0]: end = self.train_x.shape[0]
        return self.train_x[beg:end], self.train_y[beg:end], self.example_weightings[beg:end]





    def sample_val_batch(self):
        """
        Samples from the validation set
        """
        indices = npr.choice(np.arange(len(self.val_x)), self.val_batch_size, replace=False)
        return self.val_x[indices], self.val_y[indices]





    def run_epoch(self):
        accuracies, losses, recalls, precisions, aucs, msqes = [], [], [], [], [], []
        v_accuracies, v_losses, v_recalls, v_precisions, v_aucs, v_msqes = [], [], [], [], [], []
        for i in trange(self.iterations):
            '''
            getBatch() is not  a real function yet, 
            needs to be implemented to return a (input,label) pair
            the "Input" is expected to be of shape (self.batch_size, 600, 4)
            the "Label" is expected to be of shape (self.batch_size, self.num_labels)
            '''
            iter_input, iter_lbls = self.get_batch(i)
            self.run_iter(i)
            
            if self.curr_step % self.test_eval_frequency == 0:
                acc, loss, recall, precision, auc, msqe = self.eval_batch(iter_input, iter_lbls, True)

                accuracies.append(acc)
                losses.append(loss)
                recalls.append(recall)
                precisions.append(precision)
                aucs.append(auc)
                msqes.append(msqe)
        
                feed_dict = {self.x: iter_input, self.lbls: iter_lbls, self.is_train: True}
                loss_str = self.sess.run(self.train_loss_op, feed_dict=feed_dict)
                acc_str = self.sess.run(self.train_acc_op , feed_dict=feed_dict)
                prec_str = self.sess.run(self.train_prec_op, feed_dict=feed_dict)
                recall_str = self.sess.run(self.train_recall_op, feed_dict=feed_dict)
                auc_str = self.sess.run(self.train_auc_op, feed_dict=feed_dict)
                   
                self.writer.add_summary(acc_str, self.curr_step)
                self.writer.add_summary(loss_str, self.curr_step)
                self.writer.add_summary(prec_str, self.curr_step)
                self.writer.add_summary(recall_str, self.curr_step)
                self.writer.add_summary(auc_str, self.curr_step)

            if self.curr_step % self.val_eval_frequency == 0:
                acc, loss, recall, precision, auc, msqe = self.evaluate()
                v_accuracies.append(acc)
                v_losses.append(loss)
                v_recalls.append(recall)
                v_precisions.append(precision)
                v_aucs.append(auc)
                v_msqes.append(msqe)


        # TODO: some averaging over the accuracies etc if it's really unstable. E.g. reshape and then average
        return accuracies, losses, recalls, precisions, aucs, msqes, \
                v_accuracies, v_losses, v_recalls, v_precisions, v_aucs, v_msqes




    def printGraph(train, test, name):
        sns.set_style("darkgrid")
        epochs = [i+1 for i in range(self.num_epochs)]
        plt.plot(epochs, test, 'r', label = "Test")
        plt.plot(epochs, train, 'b', label = "Train")
        plt.legend(loc = "upper left")    
        plt.title(name+" vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.savefig("result_graphs/%s_%s_train_vs_test_%i_epochs.png"%(self.run_name, name, self.num_epochs))
        plt.clf()





    #CALL this Function is you want to run however many epochs you sent to the model
    def fit(self):
        train_acc, train_loss, train_recall, train_precision, train_auc, train_msqe = [],[],[],[],[],[]
        test_acc, test_loss, test_recall, test_precision, test_auc, test_msqe = [],[],[],[],[],[]

        for i in range(self.num_epochs):
            self.curr_epoch += 1
            print("-----STARTING EPOCH %i-----"%self.curr_epoch)
            acc, loss, recall, precision, auc, msqe, \
                    v_acc, v_loss, v_recall, v_precision, v_auc, v_msqe= self.run_epoch()
            train_acc.append(acc)
            train_loss.append(loss)
            train_recall.append(recall)
            train_precision.append(precision)
            train_auc.append(auc)
            train_msqe.append(msqe)
            print("-Train: ")
            print("----Accuracy: %.3f"%mean(acc))
            print("----Loss: %.3f"%mean(loss))
            print("----Recall: %.3f"%mean(recall))
            print("----Precision: %.3f"%mean(precision))
            print("----AUC: %.3f"%mean(auc))
            print("----MSQE: %0.3f"%mean(msqe))
            test_acc.append(v_acc)
            test_loss.append(v_loss)
            test_recall.append(v_recall)
            test_precision.append(v_precision)
            test_auc.append(v_auc)
            test_msqe.append(v_msqe)
            print("-Test: ")
            print("----Accuracy: %.3f"%mean(v_acc))
            print("----Loss: %.3f"%mean(v_loss))
            print("----Recall: %.3f"%mean(v_recall))
            print("----Precision: %.3f"%mean(v_precision))
            print("----AUC: %.3f"%mean(v_auc))
            print("----MSQE: %0.3f"%mean(v_msqe))
            print("---------------------------")
            if self.if_save_model:
                if self.curr_epoch % self.save_frequency == 0:
                    self.saver.save(self.sess, self.basedir+"checkpoints/%s_epoch_%i_step_%i.ckpt"%(self.run_name,self.curr_epoch, self.curr_step))

        printGraph(train_acc, test_acc, "Accuracy")
        printGraph(train_loss, test_loss, "Loss")
        printGraph(train_recall, test_recall, "Recall")
        printGraph(train_precision, test_precision, "Precision")
        printGraph(train_auc, test_auc, "Area Under ROC Curve")
        printGraph(train_msqe, test_msqe, "Mean Squared Error")

