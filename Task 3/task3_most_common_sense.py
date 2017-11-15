#!/usr/bin/env python3
"""
    task3_most_common_sense.py - Task 3: Pun Interpretation using most common sense for sense2
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/13/2017
"""

import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import treebank
from nltk.tag import hmm

def get_pun_word():
    pun_words = []
    for child in root:
        for i in range(len(child)):
            if child[i].attrib['senses'] == "2":
                pun_words.append(child[i].text.lower())
    return pun_words

def pos_for_pun_word():
    pun_pos = []
    for child in root:
        for i in range(len(child)):
            # POS tagging
            words = nltk.word_tokenize(child[i].text.lower())
            tagged_word = nltk.pos_tag(words)
    
            if child[i].attrib['senses'] == "2":
                pun_pos.append(tagged_word[0][1])

    return pun_pos

# The second sense is the most common sense extracted from WordNet'
def correct_pun_pos(pun_word, pun_pos):
    if pun_pos == 'NN' or pun_pos == 'NNS' or pun_pos == 'NNP':
        pun_pos = 'n'
    elif pun_pos == 'VB' or pun_pos == 'VBD' or pun_pos == 'VBG' or pun_pos == 'VBN' or pun_pos == 'VBP' or pun_pos == 'VBZ':
        pun_pos = 'v'
    elif pun_pos == 'JJ' or pun_pos == 'JJR' or pun_pos == 'JJS':
        pun_pos = 's'
    else:
        pun_word = porter_stemmer.stem(pun_word)
        pun_pos = nltk.pos_tag([pun_word])[0][1]
        if pun_pos == 'NN' or pun_pos == 'NNS' or pun_pos == 'NNP':
            pun_pos = 'n'
        elif pun_pos == 'VB' or pun_pos == 'VBD' or pun_pos == 'VBG' or pun_pos == 'VBN' or pun_pos == 'VBP' or pun_pos == 'VBZ':
            pun_pos = 'v'
        elif pun_pos == 'JJ' or pun_pos == 'JJR' or pun_pos == 'JJS':
            pun_pos = 's'
        else:
            pun_pos = ''

    return pun_word, pun_pos

def most_common_sense(pun_word, pun_pos):
    pun_word, pun_pos = correct_pun_pos(pun_word, pun_pos)

    if wn.synsets(pun_word, pos=pun_pos):
        most_common_synset = wn.synsets(pun_word, pos=pun_pos)[0]
        most_common_sense = most_common_synset.lemmas()[0].key()
        return most_common_sense
    else:
        # if the list of synsets is empty, try to stem the word, get a new POS, and find synsets of the new pair
        pun_word = porter_stemmer.stem(pun_word)
        pun_pos = nltk.pos_tag([pun_word])[0][1]
        pun_word, pun_pos = correct_pun_pos(pun_word, pun_pos)

        if wn.synsets(pun_word, pos=pun_pos):
            most_common_synset = wn.synsets(pun_word, pos=pun_pos)[0]
            most_common_sense = most_common_synset.lemmas()[0].key()
            return most_common_sense
        else:
            return 'none'

if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')
    
    # Load dataset from xml file (task 2)
    tree = ET.parse('../sample/subtask3-homographic-test.xml')
    root = tree.getroot()
    sentences = []
    text_ids= []
    porter_stemmer = PorterStemmer()

    for child in root:
        sentence = []
        for i in range(len(child)):
            if child[i].text.lower() not in stop_words:
                sentence.append(child[i].text.lower())
        sentences.append(sentence)

    pun_words = get_pun_word()
    pun_pos = pos_for_pun_word()

    with open('predictions/task3_sense2.txt', 'w') as file:
        for i in range(len(sentences)):
            sense = most_common_sense(pun_words[i], pun_pos[i])
            file.write(sense + '\n')
    file.close()
