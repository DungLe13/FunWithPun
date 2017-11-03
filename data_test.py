# Pun Detection Data Pre-processing
"""
    data_test.py - Data Pre-processing for Task 1: Pun Dectection
    Author: Dung Le (dungle@bennington.edu)
    Date: 10/17/2017
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

count = 0

tree = ET.parse('sample/subtask1-homographic-test.xml')
root = tree.getroot()

stop_words = set(stopwords.words('english'))
stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
#print(stop_words)

"""
for child in root:
    count += 1
    print("hom_" + str(count), end=' ')
    for i in range(len(child)):
        # POS tagging
        words = nltk.word_tokenize(child[i].text.lower())
        tagged_word = nltk.pos_tag(words)

        if tagged_word[0][0] not in stop_words:
            print(tagged_word, end=' ')
    print()
"""

