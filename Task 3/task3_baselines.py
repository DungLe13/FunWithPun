#!/usr/bin/env python3
"""
    task3_baselines.py - Task 3: Pun Interpretation choosing random word senses, or two most frequent senses
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/07/2017
"""

import xml.etree.ElementTree as ET
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

def two_random_num(syns):
    x = random.randint(0, len(syns)-1)
    y = random.randint(0, len(syns)-1)
    if x == y:
        two_random_num(syns)
    return x, y

def get_random_senses():
    with open("baselines/task3_baseline1_random.txt", "w") as file:
        for child in root:
            for i in range(len(child)):
                if child[i].attrib['senses'] == "2":
                    word_id = child[i].attrib['id']
                    word = wordnet_lemmatizer.lemmatize(child[i].text.lower())
                    synsets = wn.synsets(word)

                    # Get two senses of word
                    sense_1, sense2 = two_random_num(synsets)
                    rand_sense_1 = synsets[sense_1]
                    lemma_1 = rand_sense_1.name().split(sep='.')[0]
                    rand_sense_1 = rand_sense_1.name() + '.' + lemma_1
                    rand_sense_1_lemma = wn.lemma(rand_sense_1)
                    
                    rand_sense_2 = synsets[sense2]
                    lemma_2 = rand_sense_2.name().split(sep='.')[0]
                    rand_sense_2 = rand_sense_2.name() + '.' + lemma_2
                    rand_sense_2_lemma = wn.lemma(rand_sense_2)

                    sense_prediction = word_id + " " + rand_sense_1_lemma.key() + " " + rand_sense_2_lemma.key() + "\n"
                    file.write(sense_prediction)
    file.close()

if __name__ == "__main__":
    # Load dataset from xml file (task 3)
    tree = ET.parse('../sample/subtask3-homographic-test.xml')
    root = tree.getroot()
    wordnet_lemmatizer = WordNetLemmatizer()

    get_random_senses()
