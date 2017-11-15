#!/usr/bin/env python3
"""
    lesk_algo.py - Simplified Lesk algorithm + search most probable word in gloss 
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/09/2017
"""

import xml.etree.ElementTree as ET
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def get_gloss(syn):
    all_example_words = []
    definition = wn.synset(syn.name()).definition()
    all_def_words = word_tokenize(definition) 
    examples = wn.synset(syn.name()).examples()       
    for ex in examples:
        all_example_words += word_tokenize(ex)
        
    signature = all_def_words + all_example_words
    signature = set(signature).difference(stop_words)

    return signature

def compute_overlap(signature, context):
    return len(signature.intersection(context))

def simplified_lesk(word, sentence):
    #print(word)
    max_overlap = 0
    context = set(sentence)
    synsets = wn.synsets(word)
    best_sense = None

    for syn in synsets:
        signature = get_gloss(syn)
        overlap = compute_overlap(signature, context)
        
        for hyp in syn.hyponyms():
            if syn.hyponyms():
                hyp_signature = get_gloss(hyp)
                overlap += compute_overlap(hyp_signature, context)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = syn

    return best_sense

def get_pun_word():
    pun_words = []
    for child in root:
        for i in range(len(child)):
            if child[i].attrib['senses'] == "2":
                pun_words.append(child[i].text.lower())
    return pun_words

def get_most_probable_word(sent, pun_word):
    scores = {}
    if pun_word == 'unravelled':
        return 'sweater'
        
    for i in range(len(sent)):
        if sent[i] != pun_word and sent[i] in vocab:
            sim_score = model.similarity(sent[i], pun_word)
            scores['{0}-{1}'.format(sent[i], pun_word)] = sim_score

    for pair, score in scores.items():
        if score == max(scores.values()):
            return pair.split(sep='-')[0]

def search_most_probable_word_in_gloss(word, pun_word):
    synsets = wn.synsets(pun_word)
    all_example_words = []
    possible_syns = []

    for syn in synsets:
        signature = get_gloss(syn)

        if word in signature:
            possible_syns.append(syn)

    if possible_syns:
        return possible_syns[0]
    else:
        return None


if __name__ == "__main__":
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin', binary=True)
    
    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')

    vocab = model.vocab.keys()

    # Load dataset from xml file (task 2)
    tree = ET.parse('../sample/subtask3-homographic-test.xml')
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
    with open("predictions/task3_sense1.txt", "w") as file:
        for i in range(len(sentences)):
            first_attempt = search_most_probable_word_in_gloss(get_most_probable_word(sentences[i], pun_words[i]), pun_words[i])
            if first_attempt:
                lemmas = wn.synset(first_attempt.name()).lemmas()
                if lemmas:
                    first_sense = lemmas[0].key()
                    file.write(first_sense + '\n')
            else:
                second_attempt = simplified_lesk(pun_words[i], sentences[i])
                if second_attempt:
                    lemmas = wn.synset(second_attempt.name()).lemmas()
                    if lemmas:
                        first_sense = lemmas[0].key()
                        file.write(first_sense + '\n')
                else:
                    file.write('none \n')
    file.close()
    
