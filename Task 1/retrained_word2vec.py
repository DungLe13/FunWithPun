#!/usr/bin/env python3
"""
    retrained_word2vec.py - Retrained word2vec model using pun dataset
    Author: Dung Le (dungle@bennington.edu)
    Date: 12/01/2017
"""

import xml.etree.ElementTree as ET
import gensim
from gensim.models import Word2Vec

# Load dataset from xml file (task 1)
tree = ET.parse('../sample/subtask1-homographic-test.xml')
root = tree.getroot()

sentences = []
for child in root:
    sentence = []
    for word in child:
        sentence.append(word.text)
    sentences.append(sentence)

model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save('pun_model')
