import tensorflow as tf
from hardcnn import hardcnn

def main():
	train_path = "train/"
	test_path  = "test/"
	cell_types = ""
	logdir = "logs"
	run_name = "hard_cnn"
	model = hardcnn(run_name, train_path, test_path, cell_types, batch_size = 1000, num_epochs = 200, logdir = logdir)
	model.fit()
	
if __name__ == '__main__':
	main()