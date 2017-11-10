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
from nltk.stem.porter import PorterStemmer

def two_random_num(syns):
    if len(syns) == 2:
        x = 0
        y = 1
    else:
        x = random.randint(0, len(syns)-1)
        r = list(range(0,x)) + list(range(x+1, len(syns)-1))
        y = random.choice(r)
    return x, y

def get_random_senses():
    with open("baselines/task3_baseline1_random.txt", "w") as file:
        for child in root:
            for i in range(len(child)):
                if child[i].attrib['senses'] == "2":
                    word_id = child[i].attrib['id']
                    # special cases
                    if child[i].text.lower() == 'serves':
                        word = 'serve'
                    else:
                        word = wordnet_lemmatizer.lemmatize(child[i].text.lower())
                    synsets = wn.synsets(word)
                    if len(synsets) < 2:
                        # special case
                        if child[i].text.lower() == 'mothballed':
                            word = 'mothball'
                        else:
                            word = porter_stemmer.stem(child[i].text.lower())
                        synsets = wn.synsets(word)

                    # Get two senses of word
                    sense_1, sense_2 = two_random_num(synsets)
                    rand_sense_1 = synsets[sense_1]
                    rand_sense_1_lemmas = wn.synset(rand_sense_1.name()).lemmas()
                    rand_sense_1_lemma = rand_sense_1_lemmas[random.randint(0, len(rand_sense_1_lemmas)-1)]

                    rand_sense_2 = synsets[sense_2]
                    rand_sense_2_lemmas = wn.synset(rand_sense_2.name()).lemmas()
                    rand_sense_2_lemma = rand_sense_2_lemmas[random.randint(0, len(rand_sense_2_lemmas)-1)]

                    sense_prediction = word_id + " " + rand_sense_1_lemma.key() + " " + rand_sense_2_lemma.key() + "\n"
                    file.write(sense_prediction)
    file.close()

if __name__ == "__main__":
    # Load dataset from xml file (task 3)
    tree = ET.parse('../sample/subtask3-homographic-test.xml')
    root = tree.getroot()
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()

    get_random_senses()
