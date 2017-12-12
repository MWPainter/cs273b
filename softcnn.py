import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
import numpy.random as npr
import layers
import utils
import seaborn as sns
import time
from tqdm import trange
import matplotlib.pyplot as plt
import os
import signal

# custom SIGINT handler
ctrl_c_count = 0
def inthandler(signum, frame):
    global ctrl_c_count
    if ctrl_c_count == 1: quit()
    ctrl_c_count += 1

# Set SIGINT handler
signal.signal(signal.SIGINT, inthandler)

def mean(arr):
    return np.mean(np.array(arr))

def flatten(arr_arr):
    return [val for arr in arr_arr for val in arr]


class softcnn():

    '''
    initialiations
    '''
    def __init__(self, run_name, data_path, batch_size = 50, num_epochs = 100,  input_shape = [600,4], num_labels = 164,
                 basedir = "", logdir = "", learning_rate = 0.005, weight_decay = 0.01, if_save_model = True, save_frequency = 5, train_eval_frequency = 50, val_eval_frequency = 50, validation_iters = 10,
                 val_batch_size = 50, weight_distance_penalty = 10.0): #0.00000025):

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
        self.train_eval_frequency = train_eval_frequency
        self.val_eval_frequency = val_eval_frequency
        self.val_batch_size = val_batch_size
        self.basedir = basedir
        self.logdir = logdir
        self.weight_distance_penalty = weight_distance_penalty
        self.validation_iters = validation_iters

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
        self.avg_weighting = np.mean(self.example_weightings)

        self.learning_rate /= self.avg_weighting * 2  # account for multiplying by small numbers in the loss, by dividing by a smallish number

        self.iterations = (self.train_x.shape[0] + batch_size - 1) / batch_size

        # FOR TESTING WITH OVERFITTING:
        cell_types = self.cell_types.tolist()
        self.tasks = [cell_types.index("HPF"),
                      cell_types.index("HepG2"),
                      cell_types.index("SKIN.PEN.FRSK.KER.02"),
                      cell_types.index("PLCNT.FET"),
                      cell_types.index("PrEC"),
                      cell_types.index("HConF"),
                      cell_types.index("GM18507"),
                      cell_types.index("ESDR.H1.BMP4.MESO"),
                      cell_types.index("OVRY"),
                      cell_types.index("NHLF")]
        self.cell_types = self.cell_types[self.tasks]
        #self.train_x = self.train_x[:45000]
        self.train_y = self.train_y[:,self.tasks]#3]
        #self.val_x = self.val_x[:5000]
        self.val_y = self.val_y[:,self.tasks]#3]
        self.num_labels = 10#3
        self.class_ratios = self.compute_train_set_class_imballances(np.load(data_path + "train_y.npy")[:,self.tasks])
        self.example_weightings = self.train_y[:,:,0] * self.class_ratios + self.train_y[:,:,1] * (1 - self.class_ratios)
        self.avg_weighting = np.mean(self.example_weightings)
        self.example_weightings = 1.0 # get rid of any weightings...

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
        self.train_objective_losses = []
        self.test_objective_losses = []
        self.train_total_objective_loss = 0
        self.test_total_objective_loss = 0
        self.accuracies, self.aucs, self.msqes, self.recalls, self.precisions = [],[],[],[],[]
        self.accuracy_update_ops, self.auc_update_ops, self.msqe_update_ops, self.recall_update_ops, self.precision_update_ops = [],[],[],[],[]

        for i in range(self.num_labels):
            print("----Cell Type: " + self.cell_types[i])
            with tf.variable_scope(self.cell_types[i].replace('+','')):
                lbl = self.lbls[:,i,:]
                lbl = tf.reshape(lbl,[tf.shape(self.lbls)[0], 2])
                self.add_task_model_to_graph(lbl, i)

        
        print("----Constructing evaluation, loss, optimizaer and tensorboard ops")
        self.update_eval_metrics_ops = [self.accuracy_update_ops, self.auc_update_ops, self.msqe_update_ops, self.recall_update_ops, self.precision_update_ops]
        self.accuracy = tf.divide(tf.add_n(self.accuracies), self.num_labels)
        self.auc = tf.divide(tf.add_n(self.aucs), self.num_labels)
        self.msqe = tf.divide(tf.add_n(self.msqes), self.num_labels)
        self.recall = tf.divide(tf.add_n(self.recalls), self.num_labels)
        self.precision = tf.divide(tf.add_n(self.precisions), self.num_labels)

        self.weight_distance = self.construct_weight_distance()
        self.train_total_objective_loss /= self.num_labels
        self.total_train_loss = self.train_total_objective_loss + self.weight_distance_penalty * self.weight_distance
        self.test_total_objective_loss /= self.num_labels
        self.total_test_loss = self.test_total_objective_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = self.optimizer.minimize(self.total_train_loss)

        #TENSORBOARD operators
        self.train_loss_op = tf.summary.scalar("train_loss", self.total_train_loss)
        self.test_loss_op  = tf.summary.scalar("test_loss",  self.total_test_loss)
        self.train_acc_op = tf.summary.scalar("train_accuracy", self.accuracy)
        self.test_acc_op = tf.summary.scalar("test_accuracy", self.accuracy)
        self.train_prec_op = tf.summary.scalar("train_precision", self.precision)
        self.test_prec_op = tf.summary.scalar("test_precision", self.precision)
        self.train_recall_op = tf.summary.scalar("train_recall", self.recall)
        self.test_recall_op = tf.summary.scalar("test_recall", self.recall)
        self.train_auc_op = tf.summary.scalar("train_auc", self.auc)
        self.test_auc_op = tf.summary.scalar("test_auc", self.auc)
        self.train_msqe_op = tf.summary.scalar("train_msqe", self.msqe)
        self.test_msqe_op = tf.summary.scalar("test_msqe", self.msqe)

        self.tb_train_objective_loss_op = tf.summary.scalar("train_objective_loss", self.train_total_objective_loss)
        self.tb_test_objective_loss_op = tf.summary.scalar("test_objective_loss", self.test_total_objective_loss)
        self.tb_weight_distance_op = tf.summary.scalar("weight_distance_loss_unweighted", self.weight_distance)
        self.tb_weight_distance_loss_op = tf.summary.scalar("weight_distance_loss", self.weight_distance_penalty * self.weight_distance)
        
        weights_norm, grads_norm, grad_to_weights_ratio = self.construct_sanity_checks()
        self.tb_weights_op = tf.summary.scalar("weights_norm", weights_norm)
        self.tb_grads_op = tf.summary.scalar("gradient_norm", grads_norm)
        self.tb_grad_to_weight_ratio_op = tf.summary.scalar("grad_to_weight_ratio", grad_to_weights_ratio)
        
        print("---------------------------")





    def compute_train_set_class_imballances(self, labels):
        """
        Computes a vector of shape (self.num_labels) with the class imballances present in the dataset with labels 'labels'
        NOTE: this assumes that train_y is currently of shape (num_examples, num_tasks)
        """
        return np.mean(labels, axis=0)





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
    
        self.train_objective_losses.append(train_loss)
        self.train_total_objective_loss += train_loss
        self.test_objective_losses.append(test_loss)
        self.test_total_objective_loss += test_loss
        
        correct_labels = tf.argmax(lbls,1) # convert back from one hot
        predictions = tf.argmax(y_conv, 1)

        accuracy, accuracy_op   = tf.metrics.accuracy(correct_labels, predictions)
        recall, recall_op       = tf.metrics.recall(correct_labels, predictions)
        precision, precision_op = tf.metrics.precision(correct_labels, predictions)
        auc, auc_op             = tf.metrics.auc(correct_labels, predictions)
        msqe, msqe_op           = tf.metrics.mean_squared_error(correct_labels, predictions)

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
        conv1_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv1_weight:0')]
        conv1_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv1_bias:0')]
        conv2_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv2_weight:0')]
        conv2_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv2_bias:0')]
        conv3_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv3_weight:0')]
        conv3_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv3_bias:0')]

        total_dist = 0
        total_dist += self.pairwise_distances(conv1_weights)
        total_dist += self.pairwise_distances(conv1_biases)
        total_dist += self.pairwise_distances(conv2_weights)
        total_dist += self.pairwise_distances(conv2_biases)
        total_dist += self.pairwise_distances(conv3_weights)
        total_dist += self.pairwise_distances(conv3_biases)

        # I know it's weird to do it like this, but there's some weird typing issues that means I can't use np.prod
        pairs = 0.0
        pairs += self.param_pairs(conv1_weights)
        pairs += self.param_pairs(conv2_weights)
        pairs += self.param_pairs(conv3_weights)
        pairs += self.param_pairs(conv1_biases)
        pairs += self.param_pairs(conv2_biases)
        pairs += self.param_pairs(conv3_biases)

        return total_dist / pairs



    def param_pairs(self, arr_of_weights):
        leng = len(arr_of_weights)
        weight = arr_of_weights[0]
        params_per_var = 1
        for dim in weight.shape:
            params_per_var *= dim.value
        return params_per_var * (leng * (leng - 1) / 2)




    def pairwise_distances(self, weights):
        """
        Assumes that we are given a set of weights with identical dimensions
        Computes the sum pairwise distances between those weights
        """
        dist = 0
        for i in range(len(weights)):
            for j in range(i, len(weights)):
                diff = weights[i] if i == j else weights[i] - weights[j]
                dist += tf.nn.l2_loss(diff)
        return dist




    def construct_sanity_checks(self):
        """
        This must be called after the complete graph has been constructed

        Computes the gradient
        Computes the magnitude of the all of the weights
        Computes ops for the ratio of the two
        """
        weights_norm = tf.sqrt(tf.global_norm(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
        grads_and_var_pairs = self.optimizer.compute_gradients(self.total_train_loss)
        grads = zip(*grads_and_var_pairs)[0] 
        grads_norm = tf.sqrt(tf.global_norm(grads))

        grad_to_weights_ratio = self.learning_rate * grads_norm / weights_norm

        return weights_norm, grads_norm, grad_to_weights_ratio






    def run_iter(self, iter_number):
        """
        Runs a training step + computes loss, given a minibatch
        Runs (in parallel) the gradient computations (because we may as well not repeat work) and logs it to tb variables
        Also updates a global counter of the number of training steps that we've made
        """
        iter_input, iter_lbls, weights = self.get_batch(iter_number)
        feed_dict = {self.x: iter_input, self.lbls: iter_lbls, self.weightings: weights, self.is_train: True} 
        self.sess.run(self.train_step, feed_dict=feed_dict)
        self.curr_step += 1





    def evaluate_validation(self):
        """
        Computes evaluation metrics on the val set.  + prints summaries for tensorboard
        (Unless it was more efficient to do these ops in run_iter)
        """
        accs, losses, recalls, precisions, aucs, msqes = [], [], [], [], [], []

        for i in range(self.validation_iters):
            batch_input, batch_lbls = self.sample_val_batch()

            eval_ops = [self.accuracy, self.total_test_loss, self.recall, self.precision, self.auc, self.msqe, \
                    self.test_acc_op, self.test_loss_op, self.test_recall_op, self.test_prec_op, self.test_auc_op, self.test_msqe_op, self.tb_test_objective_loss_op]
            feed_dict = {self.x: batch_input, self.lbls: batch_lbls, self.is_train: False}

            self.sess.run(tf.local_variables_initializer())
            self.sess.run(self.update_eval_metrics_ops, feed_dict=feed_dict)
            acc, loss, recall, precision, auc, msqe, acc_str, loss_str, recall_str, prec_str, auc_str, msqe_str, obj_loss_str = self.sess.run(eval_ops, feed_dict=feed_dict)

            accs.append(acc)
            losses.append(loss)
            recalls.append(recall)
            precisions.append(precision)
            aucs.append(auc)
            msqes.append(msqe)
        
            self.writer.add_summary(acc_str, self.curr_step)
            self.writer.add_summary(loss_str, self.curr_step)
            self.writer.add_summary(recall_str, self.curr_step)
            self.writer.add_summary(prec_str, self.curr_step)
            self.writer.add_summary(auc_str, self.curr_step)
            self.writer.add_summary(msqe_str, self.curr_step)
            self.writer.add_summary(obj_loss_str, self.curr_step)

        return mean(accs), mean(losses), mean(recalls), mean(precisions), mean(aucs), mean(msqes)



    

    def eval_train_batch(self, iter_number):
        """
        Runs the evaluation operators, given a minibatch
        """
        iter_input, iter_lbls, weights = self.get_batch(iter_number)

        eval_ops = [self.accuracy, self.total_train_loss, self.recall, self.precision, self.auc, self.msqe, \
                self.train_acc_op, self.train_loss_op, self.train_recall_op, self.train_prec_op, self.train_auc_op, self.train_msqe_op, \
                self.tb_train_objective_loss_op, self.tb_weight_distance_loss_op, self.tb_weight_distance_op, self.tb_weights_op, self.tb_grads_op, self.tb_grad_to_weight_ratio_op]
        feed_dict = {self.x: iter_input, self.lbls: iter_lbls, self.weightings: weights, self.is_train: True}

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.update_eval_metrics_ops, feed_dict=feed_dict)
        acc, loss, recall, precision, auc, msqe = \
                self.sess.run([self.accuracy, self.total_train_loss, self.recall, self.precision, self.auc, self.msqe], feed_dict=feed_dict)
        acc_str, loss_str, recall_str, prec_str, auc_str, msqe_str = \
                self.sess.run([self.train_acc_op, self.train_loss_op, self.train_recall_op, self.train_prec_op, self.train_auc_op, self.train_msqe_op], feed_dict=feed_dict)
        obj_loss_str, weight_dist_loss_str, weight_dist_str, weights_str, grads_str, grad_to_weight_ratio_str = \
                self.sess.run([self.tb_train_objective_loss_op, self.tb_weight_distance_loss_op, self.tb_weight_distance_op, \
                               self.tb_weights_op, self.tb_grads_op, self.tb_grad_to_weight_ratio_op], feed_dict=feed_dict)

        #acc, loss, recall, precision, auc, msqe, acc_str, loss_str, recall_str, prec_str, auc_str, msqe_str, \
        #        obj_loss_str, weight_dist_loss_str, weight_dist_str, weights_str, grads_str, grad_to_weight_ratio_str = self.sess.run(eval_ops, feed_dict=feed_dict)
        
        self.writer.add_summary(acc_str, self.curr_step)
        self.writer.add_summary(loss_str, self.curr_step)
        self.writer.add_summary(recall_str, self.curr_step)
        self.writer.add_summary(prec_str, self.curr_step)
        self.writer.add_summary(auc_str, self.curr_step)
        self.writer.add_summary(msqe_str, self.curr_step)
        self.writer.add_summary(obj_loss_str, self.curr_step)
        self.writer.add_summary(weight_dist_loss_str, self.curr_step)
        self.writer.add_summary(weight_dist_str, self.curr_step)
        self.writer.add_summary(weights_str, self.curr_step)
        self.writer.add_summary(grads_str, self.curr_step)
        self.writer.add_summary(grad_to_weight_ratio_str, self.curr_step)

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
            if ctrl_c_count == 1: break

            self.run_iter(i)
            
            if self.curr_step % self.train_eval_frequency == 0:
                acc, loss, recall, precision, auc, msqe = self.eval_train_batch(i)
                accuracies.append(acc)
                losses.append(loss)
                recalls.append(recall)
                precisions.append(precision)
                aucs.append(auc)
                msqes.append(msqe)
        

            if self.curr_step % self.val_eval_frequency == 0:
                acc, loss, recall, precision, auc, msqe = self.evaluate_validation()
                v_accuracies.append(acc)
                v_losses.append(loss)
                v_recalls.append(recall)
                v_precisions.append(precision)
                v_aucs.append(auc)
                v_msqes.append(msqe)


        # TODO: some averaging over the accuracies etc if it's really unstable. E.g. reshape and then average
        return accuracies, losses, recalls, precisions, aucs, msqes, \
                v_accuracies, v_losses, v_recalls, v_precisions, v_aucs, v_msqes




    def printGraph(self, train, test, name):
        sns.set_style("darkgrid")
        epochs_test = [i*self.val_eval_frequency+1 for i in range(len(test))]
        epochs_train = [i*self.train_eval_frequency+1 for i in range(len(train))]
        plt.plot(epochs_test, test, 'r', label = "Test")
        plt.plot(epochs_train, train, 'b', label = "Train")
        plt.legend(loc = "upper left")    
        plt.title(name+" vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.savefig(self.basedir+"/result_graphs/%s_%s_train_vs_test_%i_epochs.png"%(self.run_name, name, self.num_epochs))
        plt.clf()





    #CALL this Function is you want to run however many epochs you sent to the model
    def fit(self):
        train_acc, train_loss, train_recall, train_precision, train_auc, train_msqe = [],[],[],[],[],[]
        test_acc, test_loss, test_recall, test_precision, test_auc, test_msqe = [],[],[],[],[],[]

        for i in range(self.num_epochs):
            if ctrl_c_count == 1: break
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

        self.printGraph(flatten(train_acc), flatten(test_acc), "Accuracy")
        self.printGraph(flatten(train_loss), flatten(test_loss), "Loss")
        self.printGraph(flatten(train_recall), flatten(test_recall), "Recall")
        self.printGraph(flatten(train_precision), flatten(test_precision), "Precision")
        self.printGraph(flatten(train_auc), flatten(test_auc), "Area Under ROC Curve")
        self.printGraph(flatten(train_msqe), flatten(test_msqe), "Mean Squared Error")

