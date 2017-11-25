import tensorflow as tf
import layers
import utils
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class hardcnn():

	'''
	initialiations
	'''
	def __init__(self, run_name, train_path, test_path, cell_types, batch_size = 1000, num_epochs = 100,  input_shape = [600,4], num_labels = 164,
		logdir = "", learning_rate = 0.002, write_to_log_frequency = 100, if_save_model = True, save_frequency = 5, iterations = 10):
		self.input_shape = input_shape
		self.train_path  = train_path
		self.test_path   = test_path
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.num_labels = num_labels
		self.cell_types = cell_types
		self.curr_step  = 0
		self.curr_epoch = 0
		self.num_epochs = num_epochs
		self.write_to_log_frequency = write_to_log_frequency
		self.run_name = run_name
		self.if_save_model = if_save_model
		self.save_frequency = save_frequency
		self.iterations = iterations

		#sets up the new tf session
		tf.reset_default_graph()		
		self.sess = tf.Session()
		
		#sets up the graph of the network
		with tf.variable_scope("hard_cnn"):
			self.make_graph()

		#sets up the writer for tensorboard functionality
		if not os.path.isdir(logdir):
			os.mkdir(logdir)

		dir_path = os.path.dirname(os.path.realpath(__file__))
		if not os.path.isdir(dir_path+"/hard_model/result_graphs"):
			os.mkdir(dir_path+"/hard_model/result_graphs")

		if not os.path.isdir(dir_path+"/checkpoints"):
			os.mkdir(dir_path+"/checkpoints")

		self.writer = tf.summary.FileWriter(logdir,self.sess.graph)
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

	def make_graph(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size,self.input_shape[0], self.input_shape[1]])
		self.lbls = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_labels])
		self.is_train = tf.placeholder(tf.bool)


		self.accuracy, self.precision, self.recall, self.auc, self.msqe, self.total_loss, self.y_conv, self.predictions = self.make_single_graph(self.x, self.lbls)
		self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)

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
		self.saver = tf.train.Saver()

	def make_single_graph(self, x, labels):

		conv1 =  layers.max_pool1d(
					layers.batch_norm(
						layers.conv1dLayer(x, filterSize = 19, outputDim = 300, stride = 1, name = "conv1"),
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

		y_conv = tf.nn.sigmoid(layers.readoutLayer(fc2, outputDim = 164, name = "readout"))
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = y_conv))

		predictions = tf.to_int32(y_conv > 0.5)

		#this part is currently not working, computing metrics below directly based on predictions, 
		#rather than tensors
		accuracy = tf.Variable(0.0)
		recall = tf.Variable(0.0)
		precision = tf.Variable(0.0)
		auc = tf.Variable(0.0)
		msqe = tf.Variable(0.0)
		for i in range(self.num_labels):
			accuracy += tf.reduce_mean(tf.to_float(predictions[:,i] == labels[:,i]))
			recall += tf.metrics.recall(labels[:,i], predictions[:,i])[0]
			precision += tf.metrics.precision(labels[:,i], predictions[:,i])[0]
			auc += tf.metrics.auc(labels[:,i], predictions[:,i])[0]
			msqe += tf.metrics.mean_squared_error(labels[:,i], y_conv[:,i])[0]

		accuracy = tf.divide(accuracy, self.num_labels)
		recall = tf.divide(recall, self.num_labels)
		precision = tf.divide(precision, self.num_labels)
		auc = tf.divide(auc, self.num_labels)
		msqe = tf.divide(msqe, self.num_labels)
		return accuracy, precision, recall, auc, msqe, loss, y_conv, predictions

	def get_metrics(self, predictions, labels, y_conv):
		acc, recall, precision, auc_score, msqe = 0, 0, 0, 0, 0
		for i in range(self.num_labels):
			acc += accuracy_score(predictions[:,i], labels[:,i])
			recall += recall_score(predictions[:,i], labels[:,i])
			precision += precision_score(predictions[:,i], labels[:,i])
			try:
				auc_score += roc_auc_score(predictions[:,i], labels[:,i])
			except ValueError:
				pass

		msqe = mean_squared_error(labels, y_conv)
		auc_score  /= self.num_labels
		acc  /= self.num_labels
		precision  /= self.num_labels
		recall /= self.num_labels
		return acc, recall, precision, auc_score, msqe

	def run_iter(self, iter_input, iter_lbls, to_log, is_train = True):
		# g = tf.Graph()
		# with g.as_default():
		if is_train:
			self.sess.run(self.train_step, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
			self.curr_step += 1

		# acc = self.sess.run(self.accuracy, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		# recall = self.sess.run(self.recall, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		# precision = self.sess.run(self.precision, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		# auc = self.sess.run(self.auc, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		# msqe = self.sess.run(self.msqe, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})

		loss = self.sess.run(self.total_loss, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		y_conv = self.sess.run(self.y_conv, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		predictions = self.sess.run(self.predictions, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
		acc, recall, precision, auc_score, msqe = self.get_metrics(predictions, iter_lbls, y_conv)

		if to_log:
			if is_train:
				loss_str = self.sess.run(self.train_loss_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				acc_str = self.sess.run(self.train_acc_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train}) 
				prec_str = self.sess.run(self.train_prec_op , feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				recall_str = self.sess.run(self.train_recall_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				auc_str = self.sess.run(self.train_auc_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
			else:
				loss_str = self.sess.run(self.test_loss_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				acc_str = self.sess.run(self.test_acc_op , feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				prec_str = self.sess.run(self.test_prec_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				recall_str = self.sess.run(self.test_recall_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
				auc_str = self.sess.run(self.test_auc_op, feed_dict={self.x: iter_input, self.lbls: iter_lbls, self.is_train: is_train})
			self.writer.add_summary(acc_str, self.curr_step)
			self.writer.add_summary(loss_str, self.curr_step)
			self.writer.add_summary(prec_str, self.curr_step)
			self.writer.add_summary(recall_str, self.curr_step)
			self.writer.add_summary(auc_str, self.curr_step)

		return acc, loss, recall, precision, auc_score, msqe


	def get_batch(self, batch_size, is_train):
		inputs, labels = [], []
		if is_train:
			features = np.load("train_x.npy")
			labels = np.load("train_y.npy")
			indices = np.random.choice(np.arange(len(features)), batch_size, replace=False)
		else:
			features = np.load("val_x.npy")
			labels = np.load("val_y.npy")
			indices = np.random.choice(np.arange(len(features)), batch_size, replace=False)
		inputs = features[indices]
		labels = labels[indices]

		return inputs, labels

	def run_epoch(self, is_train = True):
		accuracies, losses, recalls, precisions, auces, msqes = [], [], [], [], [], []
		for i in range(self.iterations):

			iter_input, iter_lbls = self.get_batch(self.batch_size, is_train)
			to_log = False
			if i % self.write_to_log_frequency == 0: to_log = True
			acc, loss, recall, precision, auc, msqe = self.run_iter(iter_input, iter_lbls, to_log, is_train)
			accuracies.append(acc)
			losses.append(loss)
			recalls.append(recall)
			precisions.append(precision)
			auces.append(auc)
			msqes.append(msqe)

		avg_acc  = float(sum(accuracies))/float(max(len(accuracies),1))
		avg_loss = float(sum(losses))/float(max(len(losses),1))
		avg_recall = float(sum(recalls))/float(max(len(recalls),1))
		avg_precision = float(sum(precisions))/float(max(len(precisions),1))
		avg_auc = float(sum(auces))/float(max(len(auces),1))
		avg_msqe = float(sum(msqes))/float(max(len(msqes),1))

		return avg_acc, avg_loss, avg_recall, avg_precision, avg_auc, avg_msqe

	def printGraph(self, train, test, name):
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
			acc, loss, recall, precision, auc, msqe = self.run_epoch(is_train = True)
			train_acc.append(acc)
			train_loss.append(loss)
			train_recall.append(recall)
			train_precision.append(precision)
			train_auc.append(auc)
			train_msqe.append(msqe)
			print("-Train: ")
			print("----Accuracy: %.3f"%acc)
			print("----Loss: %.3f"%loss)
			print("----Recall: %.3f"%recall)
			print("----Precision: %.3f"%precision)
			print("----AUC: %.3f"%auc)
			print("----MSQE: %0.3f"%msqe)
			acc, loss, recall, precision, auc, msqe = self.run_epoch(is_train = False)
			test_acc.append(acc)
			test_loss.append(loss)
			test_recall.append(recall)
			test_precision.append(precision)
			test_auc.append(auc)
			test_msqe.append(msqe)
			print("-Test: ")
			print("----Accuracy: %.3f"%acc)
			print("----Loss: %.3f"%loss)
			print("----Recall: %.3f"%recall)
			print("----Precision: %.3f"%precision)
			print("----AUC: %.3f"%auc)
			print("----MSQE: %0.3f"%msqe)
			print("---------------------------")
			if self.if_save_model:
				if self.curr_epoch % self.save_frequency == 0:
					self.saver.save(self.sess,"checkpoints/%s_epoch_%i_step_%i.ckpt"%(self.run_name,self.curr_epoch, self.curr_step))
			
		self.printGraph(train_acc, test_acc, "Accuracy")
		self.printGraph(train_loss, test_loss, "Loss")
		self.printGraph(train_recall, test_recall, "Recall")
		self.printGraph(train_precision, test_precision, "Precision")
		self.printGraph(train_auc, test_auc, "Area Under ROC Curve")
		self.printGraph(train_msqe, test_msqe, "Mean Squared Error")
