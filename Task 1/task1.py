#!/usr/bin/env python3
"""
    task1.py - Task 1: Pun Detection using word2vec and RNN with LSTM cells
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/13/2017
"""

import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from data_processing import get_train_and_test_data

train_x, train_y, test_x, test_y = get_train_and_test_data()
train_data = train_x, train_y

'''
I will want the train_x and text_x to be a gigantic numpy array,
i.e. array([__sentence_1_word_1_300__, __sentence_2_word_1_300__, ..., __sentence_n_word_1_300__])
num_step will be the length for each sentence (varies between sentence - will not be static)
this will mess up with other arrays and matrix too.
Need solution
- probably put each num_step for each sentence in an array
- maybe for each sentence, choose a certain number of words = num_step for every sentence
'''

# Load dataset from xml file (task 1)
tree1 = ET.parse('../sample/subtask1-homographic-test.xml')
root1 = tree1.getroot()
original_sentences = []
text_ids= []

for child in root1:
    original_sentence = []
    text_id = child.attrib['id']
    for i in range(len(child)):
        original_sentence.append(child[i].text.lower())
    original_sentences.append(original_sentence)
    text_ids.append(text_id)

# Global config variables
num_steps = 3 # number of truncated backprop steps ('n' in the discussion above)
'''
for i in range(len(train_x)):
    num_steps = len(train_x[i])
'''
# Batch_size will be 300 because each word is represented as a 300 dimensional vector
batch_size = 100
# Number of output classes, 2 in this case (0 and 1 only)
num_classes = 2
# state_size is how many layers (hidden + output layers), 4 in this case means 3 hidden layers + 1 output layer
state_size = 4
learning_rate = 0.1

def gen_batch(raw_data, batch_size, num_steps):
    data_length = 300

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    print(batch_partition_length)  # should equal 3
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)

    for j in range(len(train_x)):
        for i in range(batch_size):
            print(train_x[j][batch_partition_length * i:batch_partition_length * (i + 1)])
            data_x[i] = train_x[j][batch_partition_length * i:batch_partition_length * (i + 1)]
            data_y[i] = train_y[j][batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps
    print(epoch_size)

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(train_data, batch_size, num_steps)

'''
=========================================================================================
'''

"""
Placeholders
"""
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)

"""
Inputs
"""
rnn_inputs = tf.one_hot(x, num_classes)
print(rnn_inputs)

"""
RNN
"""
cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
#rnn_outputs = tf.Print(rnn_outputs, [rnn_outputs])

"""
Predictions, loss, training step
"""
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)
print(predictions)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

"""
Train the network
"""
def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())                                                                                                                                                                                     
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            #training_state = np.zeros((batch_size, state_size))
            _current_cell_state = np.zeros((batch_size, state_size))
            _current_hidden_state = np.zeros((batch_size, state_size))
            
            if verbose:
                print("\nEPOCH", idx)
            
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = sess.run(
                    [losses, total_loss, final_state, train_step],
                    feed_dict = {
                        x: X,
                        y: Y,
                        cell_state: _current_cell_state,
                        hidden_state: _current_hidden_state
                    })
                _current_cell_state, _current_hidden_state = training_state
                training_loss += training_loss_
                
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step, "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses

training_losses = train_network(3, num_steps)
