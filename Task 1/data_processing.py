#!/usr/bin/env python3
"""
    data_processing.py - Task 1: Pun Detection using word2vec and RNN with LSTM cells
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/7/2017
"""

import xml.etree.ElementTree as ET
import gensim
import tensorflow as tf
import numpy as np
from retrained_word2vec import sentences

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin', binary=True)

# Load retrained model (trained on pun data)
# model = gensim.models.Word2Vec.load('pun_model')

# Load dataset from xml file (task 1)
tree1 = ET.parse('../sample/subtask1-homographic-test.xml')
root1 = tree1.getroot()

# Load dataset from xml file (task 2)
tree2 = ET.parse('../sample/subtask2-homographic-test.xml')
root2 = tree2.getroot()

original_sentences = []
text_ids= []
pun_sentences = []

# input_x contains all the vector respresentation of a sentence
# each element of input_x is a tensor variable with the shape of (a, 300)
# with a being the number of words (in vocab) of each sentence
input_x = []
# output_y contains all the classification for each sentence, with 0 = non-pun and 1 = pun
output_y = []

for text in root2.iter('text'):
    pun_sentences.append(text.attrib['id'])

for i in range(1, 2251):
    sent = 'hom_' + str(i)
    if sent in pun_sentences:
        output_y.append(1)
    else:
        output_y.append(0)

for child in root1:
    original_sentence = []
    text_id = child.attrib['id']
    for i in range(len(child)):
        original_sentence.append(child[i].text.lower())
    original_sentences.append(original_sentence)
    text_ids.append(text_id)

for i in range(len(original_sentences)):
    # Input x for one sentence: containing all the word vectors of the sentence
    sent_list = []
    for w in original_sentences[i]:
        try:
            word_vec = model.wv[w].tolist()
            sent_list.append(word_vec)
        except KeyError:
            pass
            
    input_x.append(sent_list)
    
def get_train_and_test_data(test_size=0.2):
    testing_size = int(test_size * len(original_sentences))

    train_x = list(input_x[:-testing_size])
    train_y = list(output_y[:-testing_size])

    test_x = list(input_x[-testing_size:])
    test_y = list(output_y[-testing_size:])

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = get_train_and_test_data()

