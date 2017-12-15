#!/usr/bin/env python
from optparse import OptionParser
import sys
import numpy as np
import numpy.random as npr
import dna_io

"""
Bit of a weird preprocessing script
Data file should be of fasta format, but with all ")\n" strings replaced by ") " and all ">xx" replaced by ""
Label file the activations as outputted by basset
"""


# Get file names
usage = "usage: %prog [options] <fasta file> <targets file> <target_labels_out> "\
        "<train_x out> <train_y out> <val_x out> <val_y out> <test_x out> <test_y out>" 
parser = OptionParser(usage)
parser.add_option('-t', dest='test_set_size', default=0, type=int, help='Size of the test set')
parser.add_option('-v', dest='val_set_size', default=0, type=int, help='Size of the validation set')
parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [default: %default]')
parser.add_option('-s', dest='random_seed', default=42, type='int', help='numpy.random see [Default: %default]')
(options, args) = parser.parse_args()

if len(args) != 9:
    parser.error('Must provide all filenames required')
    parser.print_help()
    quit()

# filenames
fasta_file = args[0]
targets_file = args[1]
target_labels_file = args[2]
train_x_file = args[3]
train_y_file = args[4]
val_x_file = args[5]
val_y_file = args[6]
test_x_file = args[7]
test_y_file = args[8]


# get data
print "getting data"
seqs, targets = dna_io.load_data_1hot(fasta_file, targets_file, extend_len=None, mean_norm=False, \
                                      whiten=False, permute=False, sort=False)


assert(seqs.shape[0] == targets.shape[0])

seqs = seqs.reshape((seqs.shape[0], 4, seqs.shape[1]/4)) # shape = (dataset_size, 4, 600)
seqs = np.transpose(seqs, (0,2,1)) # make shape = (dataset_size, 600, 4)

# get an array of the cell types
print "getting target labels"
target_labels = []
with open(targets_file, "r") as target_file:
    target_labels = target_file.readline().strip().split("\t")


# permute data if need be
if options.permute:
    print "permuting"
    npr.seed(options.random_seed)
    order = npr.permutation(seqs.shape[0])
    seqs = seqs[order]
    targets = targets[order]


# split
print "splitting data"
val_num = options.val_set_size
test_num = options.test_set_size
train_num = seqs.shape[0] - val_num - test_num

train_val_div = train_num
val_test_div = train_num + val_num

train_x = seqs[:train_val_div]
train_y = targets[:train_val_div]
val_x = seqs[train_val_div:val_test_div]
val_y = targets[train_val_div:val_test_div]
test_x = seqs[val_test_div:]
test_y = targets[val_test_div:]


# Output some stuff to validate
print "Train_x shape: ", train_x.shape
print "Train_y shape: ", train_y.shape
print "Val_x shape: ", val_x.shape
print "Val_y shape: ", val_y.shape
print "Test_x shape: ", test_x.shape
print "Test_y shape: ", test_y.shape

# write to files
print "writing split sets to files"
np.save(target_labels_file, np.array(target_labels))
np.save(train_x_file, train_x)
np.save(train_y_file, train_y)
np.save(val_x_file, val_x)
np.save(val_y_file, val_y)
np.save(test_x_file, test_x)
np.save(test_y_file, test_y)
