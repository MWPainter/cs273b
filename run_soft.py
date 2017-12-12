import tensorflow as tf
from softcnn import softcnn

def main():
    run_name = "soft_small_set_0"
    data_path = "/mnt/data/"
    basedir = "/mnt/" + run_name + "/"
    logdir = basedir + "logs/"
    model = softcnn(run_name, data_path, batch_size = 50, num_epochs = 100, weight_decay = 0.01, basedir = basedir, logdir = logdir)
    model.fit()
    

if __name__ == '__main__':
    main()
