import tensorflow as tf
from softcnn import softcnn

def main():
	data_path = "/mnt/data/"
	logdir = ""
	run_name = ""
	model = softcnn(run_name, data_path, cell_types, batch_size = 50, num_epochs = 100, weight_decay = 0.01, logdir = "")
	model.fit()
	

if __name__ == '__main__':
	main()
