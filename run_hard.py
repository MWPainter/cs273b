import tensorflow as tf
from hardcnn import hardcnn

def main():
    run_name = "hard_run_weighted_0"
    data_path = "/mnt/data/"
    basedir = "/mnt/" + run_name + "/"
    logdir = basedir + "logs/"
    model = hardcnn(run_name, data_path, batch_size = 50, num_epochs = 100, basedir = basedir, logdir = logdir)
    model.fit()
    

if __name__ == '__main__':
    main()
