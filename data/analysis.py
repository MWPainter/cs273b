import numpy as np

labels = np.load('target_labels.npy')
train_y = np.load('train_y.npy')
val_y = np.load('val_y.npy') 
test_y = np.load('test_y.npy')

acc_train = np.sum(train_y, axis=1)
acc_val = np.sum(val_y, axis=1)
acc_test = np.sum(test_y, axis=1)
rat_train = acc_train / float(train_y.shape[0])
rat_val = acc_val / float(val_y.shape[0])
rat_test = acc_test / float(test_y.shape[0])

print "acc = accessbile, rat = ratio of accessible examples"
print ""
print "total training examples: ", train_y.shape[0]
print "total validation examples: ", val_y.shape[0]
print "total test examples: ", test_y.shape[0]

print "target_labels".rjust(30), "train_acc".rjust(10), "train_rat".rjust(15), \
        "val_acc".rjust(10), "val_rat".rjust(15), "test_acc".rjust(10), "test_rat".rjust(15)

for i in xrange(labels.shape[0]):
    print labels[i].rjust(30), str(acc_train[i]).rjust(10), ("%.8f" % rat_train[i]).rjust(15), \
            str(acc_val[i]).rjust(10), ("%.8f" % rat_val[i]).rjust(15), \
            str(acc_test[i]).rjust(10), ("%.8f" % rat_test[i]).rjust(15)

