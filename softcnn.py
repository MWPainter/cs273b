import tensorflow as tf
import layers
import utils


class softcnn():

	def __init__(self, train_path, test_path, cell_types, batch_size = 50, num_epochs = 100,  input_shape = [600,4], num_labels = 164,
		logdir = "", , learning_rate = 0.002, write_to_log_frequency = 100, weight_decay = 0.01):
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
		self.weight_decay = weight_decay
		self.write_to_log_frequency = write_to_log_frequency

		tf.reset_default_graph()		
		self.sess = tf.Session()
		
		with tf.variable_scope("soft_cnn"):
			self.make_graph()

		if not os.path.isdir(logdir):
			os.mkdir(logdir)
		self.writer = tf.summary.FileWriter(logdir,self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

	def make_graph(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size,self.input_shape[0], self.input_shape[1], 1])
		self.lbls = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_labels])
		self.accuracies, self.precision, self.auces, self.msqes, self.losses, self.recalls, self.precisions = [],[],[],[],[],[],[]
		self.total_loss = None
		for i in range(self.num_labels):
			with tf.variable_scope(self.cell_types[i]):
				acc, loss, recall, precision, auc, msqe = self.make_single_graph(self.x, self.lbls, 1)


			self.losses.append(loss)
			self.accuracies.append(acc)
			self.msqes.append(msqe)
			self.auces.append(auc)
			self.total_loss += loss

		self.accuracy = tf.divide(tf.add_n(self.accuracies), self.num_labels)
		self.auc = tf.divide(tf.add_n(self.auces), self.num_labels)
		self.msqe = tf.divide(tf.add_n(self.msqes), self.num_labels)
		self.recall = tf.divide(tf.add_n(self.recalls), self.num_labels)
		self.precision = tf.divide(tf.add_n(self.precisions), self.num_labels)


		weightVars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('conv1_weights:0') or v.name.endswith('conv2_weights:0') or v.name.endswith('conv3_weights:0')]
		self.l2Reg = tf.add_n([tf.nn.l2_loss(w) for w in weightVars])*self.weight_decay
		self.total_loss /= self.num_labels
		self.total_loss += self.l2Reg

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

	def make_single_graph(self, x, lbls, number_of_final_inputs):


		conv1 =  layers.max_pool(
					layers.convLayer(x, filterSize = [19,4], outputDim = 300, strides = [1,1,1,1], name = "conv1"),
					, size = 4, strides = [1,4,4,1], name = "pool1")

		conv2 =  layers.max_pool(
					layers.convLayer(self.conv1, filterSize = [11,4], outputDim = 200, strides = [1,1,1,1], name = "conv2")
					, size = 4, strides = [1,4,4,1], name = "pool2")

		conv3 =  layers.max_pool(
					layers.convLayer(self.conv2, filterSize = [7,4], outputDim = 200, strides = [1,1,1,1], name = "conv3")
					, size = 4, strides = [1,4,4,1], name = "pool3")

		fc1   = layers.denseLayer(self.conv3, outputDim = 1000, name = "fc1")
		fc2   = layers.denseLayer(self.fc1, outputDim = 1000, name = "fc2")

		y_conv = layers.readoutLayer(self.fc2, outputDim = number_of_final_inputs, name = "readout")
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = lbls, logits = y_conv))
		# train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

		predictions = tf.argmax(y_conv, 1)

		accuracy    = tf.metrics.accuracy(tf.argmax(lbls,1), predictions)
		recall      = tf.metrics.recall(tf.argmax(lbls,1), predictions)
		precision   = tf.metrics.precision(tf.argmax(lbls,1), predictions)
		auc         = tf.metrics.precision(tf.argmax(lbls,1), predictions)
		msqe        = tf.metrics.mean_squared_error(tf.argmax(lbls,1), predictions)


		return accuracy, loss, recall, precision, auc, msqe


		# self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		
		# self.prediction = tf.argmax(self.y_conv, 1)
		# self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.lbls, 1))		
		# self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# self.train_acc_op = tf.summary.scalar("train_accuracy", self.accuracy)
		# self.test_acc_op = tf.summary.scalar("test_accuracy", self.accuracy)


	def run_iter(self, iter_input, iter_lbls, to_log, is_train = True):
		if is_train:
			self.sess.run(self.train_step, self.x = , self.lbls =  )
			self.curr_step += 1

		acc = self.sess.run(self.accuracy, feed_dict={self.x = iter_input, self.lbls = iter_lbls})
		loss = self.sess.run(self.total_loss, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
		recall = self.sess.run(self.recall, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
		precision = self.sess.run(self.precision, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
		auc = self.sess.run(self.auc, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
		msqe = self.sess.run(self.msqe, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))

		if to_log:
			if is_train:
				loss_str = self.sess.run(self.train_loss_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				acc_str = self.sess.run(self.train_acc_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls)) 
				prec_str = self.sess.run(self.train_prec_op , feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				recall_str = self.sess.run(self.train_recall_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				auc_str = self.sess.run(self.train_auc_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
			else:
				loss_str = self.sess.run(self.test_loss_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				acc_str = self.sess.run(self.test_acc_op , feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				prec_str = self.sess.run(self.test_prec_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				recall_str = self.sess.run(self.test_recall_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
				auc_str = self.sess.run(self.test_auc_op, feed_dict=(self.x = iter_input, self.lbls = iter_lbls))
			self.writer.add_summary(acc_str, self.curr_step)
			self.writer.add_summary(loss_str, self.curr_step)
			self.writer.add_summary(prec_str, self.curr_step)
			self.writer.add_summary(recall_str, self.curr_step)
			self.writer.add_summary(auc_str, self.curr_step)

		return acc, loss, recall, precision, auc, msqe


	def run_epoch(self, is_train = True):
		accuracies, losses, recalls, precisions, auces, msqes = [], [], [], [], [], []
		for i in range(self.iterations):
			'''
			getBatch() is not  a real function yet, 
			needs to be implemented to return a (input,label) pair
			the "Input" is expected to be of shape (self.batch_size, 600, 4)
			the "Label" is expected to be of shape (self.batch_size, self.num_labels)
			'''
			iter_input, iter_lbls = getBatch(self.batch_size)
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


	#CALL this Function is you want to run however many epochs you sent to the model
	def fit(self):
		for i in range(num_epochs):
			self.curr_epoch += 1
			print("-----STARTING EPOCH %i-----"%self.curr_epoch)
			acc, loss, recall, precision, auc, msqe = self.run_epoch(is_train = True)
			print("-Train: ")
			print("----Accuracy: %.3f"%acc)
			print("----Loss: %.3f"%loss)
			print("----Recall: %.3f"%recall)
			print("----Precision: %.3f"%precision)
			print("----AUC: %.3f"%auc)
			print("----MSQE: %0.3f"%msqe)
			acc, loss, recall, precision, auc, msqe = self.run_epoch(is_train = False)
			print("-Test: ")
			print("----Accuracy: %.3f"%acc)
			print("----Loss: %.3f"%loss)
			print("----Recall: %.3f"%recall)
			print("----Precision: %.3f"%precision)
			print("----AUC: %.3f"%auc)
			print("----MSQE: %0.3f"%msqe)
			print("---------------------------")
