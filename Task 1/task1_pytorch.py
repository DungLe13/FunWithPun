#!/usr/bin/env python3
"""
    task1_pytorch.py - Task 1: Pun Detection using word2vec and RNN with LSTM cells (PyTorch)
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/30/2017
"""

import xml.etree.ElementTree as ET
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_processing import get_train_and_test_data

train_x, train_y, test_x, test_y = get_train_and_test_data()
"""
Note to self:
1. train_x and train_y both contain 1800 lists = 1800 training samples
2. train_x[0] = the first element of the list, that looks something like
   [[__300 features of 1st word__], [__300 features of 2nd word__], ..., [__300 features of nth word__]]
   with n being the length of each sentence.
3. train_y is a list constitutes of 0s and 1s
"""

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

"""
    RECURRENT NEURAL NETWORK with LONG-SHORT TERM MEMORY CELLS
"""
# PARAMETERS
# The dimension of word embeddings, which is 300 (features)
EMBEDDING_DIM = 300
# hidden_dim = dimension of hidden layers
HIDDEN_DIM = 40

# PREPARING FOR THE INPUT AND OUPUT SEQUENCE
train_x_seq = []
for sent_vec in train_x:
    sent_tensor = torch.Tensor(sent_vec)
    input_sent = sent_tensor.view(len(sent_vec), 1, EMBEDDING_DIM)
    train_x_seq.append(input_sent)

targets = Variable(torch.Tensor(train_y))
# print(targets[0])
# print(train_x_seq[0])
# print(len(train_x_seq[0]))

# MODEL
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_classes=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.hidden2classifier = nn.Linear(hidden_dim, num_classes)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, embeds):
        lstm_out, self.hidden = self.lstm(Variable(embeds), self.hidden)
        classify_space = self.hidden2classifier(lstm_out.view(len(embeds), -1))
        classify_scores = F.softmax(classify_space)
        return classify_scores[-1]

# TRAINING AND TESTING MODEL
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
"""
print("============ BEFORE ===============")
inputs = train_x_seq[0]
last_score = model(inputs)
print(last_score)
"""
for epoch in range(5):
    for i in range(len(train_x_seq)):
        # STEP 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # STEP 2. Since inputs are in the right shape, run the forward propagation
        classify_score = model(train_x_seq[i])
        # binary_score = (classify_score[0] > classify_score[1]).float()
        # print(binary_score)

        # STEP 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(classify_score, targets[i])
        loss.backward()
        optimizer.step()
"""
print("============ AFTER ===============")
inputs = train_x_seq[0]
last_score = model(inputs)
print(last_score)
"""
train_results = []
for sent_tensor in train_x_seq:
    last_score = model(sent_tensor)
    train_results.append(last_score.data[0])

print(train_results)
