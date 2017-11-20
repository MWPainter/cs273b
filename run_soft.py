import tensorflow as tf
from softcnn import softcnn

def main():
	train_path = ""
	test_path  = ""
	cell_types = ""
	logdir = ""
	model = softcnn(train_path, test_path, cell_types,batch_size = 50, num_epochs = 100, weight_decay = 0.01, logdir = "")
	model.fit()
	

if __name__ == '__main__':
	main()