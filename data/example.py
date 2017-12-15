import numpy as np

labels = np.load('target_labels.npy')
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
val_x = np.load('val_x.npy') 
val_y = np.load('val_y.npy') 
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')

print "Labels = ", labels
print ""

print "train_x.shape = ", train_x.shape
print "train_x[0] = ", train_x[0]
print ""

print "train_y.shape = ", train_y.shape
print "train_y[0] = ", train_y[0]
print ""

print "val_x.shape = ", val_x.shape
print "val_x[0] = ", val_x[0]
print ""

print "val_y.shape = ", val_y.shape
print "val_y[0] = ", val_y[0]
print ""

print "test_x.shape = ", test_x.shape
print "test_x[0] = ", test_x[0]
print ""

print "test_y.shape = ", test_y.shape
print "test_y[0] = ", test_y[0]
print ""

