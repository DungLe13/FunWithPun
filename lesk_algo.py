#!/usr/bin/env python3
"""
    lesk_algo.py - Simplified Lesk algorithm
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/09/2017
"""

import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

def compute_overlap(signature, context):
    return len(signature.intersection(context))

def simplified_lesk(word, sentence):
    max_overlap = 0
    context = set(sentence)
    synsets = wn.synsets(word)
    print(len(synsets))
    most_common_synset = synsets[0]
    print(len(most_common_synset.lemmas()))
    most_common_sense = most_common_synset.lemmas()[0].key()

    all_senses = []
    all_senses_lemmas = wn.lemmas(word)
    for lemma in all_senses_lemmas:
        all_senses.append(lemma.key())

    all_def_words = []
    all_example_words = []
    for syn in synsets:
        definition = wn.synset(syn.name()).definition()
        all_def_words += definition.split(' ')
        examples = wn.synset(syn.name()).examples()       
        for ex in examples:
            words = ex.split(' ')
            all_example_words += words

    print(all_senses)
    print(all_def_words)
    print(all_example_words)
    for sense in all_senses:
        signature = set(all_def_words + all_example_words)
        signature = signature.difference(stop_words)
        overlap = compute_overlap(signature, context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense

def get_pun_word():
    pun_words = []
    for child in root:
        for i in range(len(child)):
            if child[i].attrib['senses'] == "2":
                pun_words.append(child[i].text.lower())
    return pun_words

if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')

    # Load dataset from xml file (task 2)
    tree = ET.parse('sample/subtask3-homographic-test.xml')
    root = tree.getroot()
    sentences = []
    text_ids= []

    for child in root:
        sentence = []
        for i in range(len(child)):
            if child[i].text.lower() not in stop_words:
                sentence.append(child[i].text.lower())
        sentences.append(sentence)

    pun_words = get_pun_word()
    for i in range(len(sentences)):
        print(simplified_lesk(pun_words[i], sentences[i]))
    
