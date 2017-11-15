# Pun Detection Data Pre-processing
"""
    data_test.py - Data Pre-processing for Task 1: Pun Dectection
    Author: Dung Le (dungle@bennington.edu)
    Date: 10/17/2017
"""

import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords

from nltk.tag.stanford import StanfordPOSTagger

count = 0
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

tree = ET.parse('sample/subtask1-homographic-test.xml')
root = tree.getroot()

stop_words = set(stopwords.words('english'))
stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
#print(stop_words)

for child in root:
    count += 1
    print("hom_" + str(count), end=' ')
    for i in range(len(child)):
        # POS tagging
        words = nltk.word_tokenize(child[i].text.lower())
        tagged_word = st.tag(words)

        if tagged_word[0][0] not in stop_words:
            print(tagged_word, end=' ')
    print()

