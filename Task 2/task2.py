#!/usr/bin/env python3
"""
    task2.py - Task 2: Pun Location using word2vec and cosine similarities
    Author: Dung Le (dungle@bennington.edu)
    Date: 10/17/2017
"""

import xml.etree.ElementTree as ET
import gensim
import pickle
import nltk
from nltk.corpus import stopwords

def get_possible_pun_words(sent):
    scores = {}
    if len(sent) <= 1:
        return set(sent[0])
    else:
        for i in range(len(sent)-1):
            for j in range(i+1, len(sent)):
                sim_score = model.similarity(sent[i], sent[j])
                scores['{0}-{1}'.format(sent[i], sent[j])] = sim_score

        if len(scores) >= 5:
            top2 = sorted(zip(scores.values(), scores.keys()), reverse=True)[:2]
            poss = [tup[1].split(sep='-') for tup in top2]
            possible_pun_words = set(poss[0] + poss[1])
        else:
            #poss = [pair.split(sep='-') for pair in scores.keys()]
            #possible_pun_words = set()
            #for i in range(len(poss)):
            #    possible_pun_words = possible_pun_words.union(set(poss[i]))
            top = sorted(zip(scores.values(), scores.keys()), reverse=True)[:1]
            poss = [tup[1].split(sep='-') for tup in top]
            possible_pun_words = set(poss[0])
            
        return possible_pun_words

def get_pun_word(orig_sent, sent):
    pun_words = get_possible_pun_words(sent)
    largest_index = 0

    for w in pun_words:
        index = orig_sent.index(w)
        if index > largest_index:
            largest_index = index

    return largest_index + 1

if __name__ == "__main__":
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin', binary=True)
    
    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')

    # Load dataset from xml file (task 2)
    tree = ET.parse('../sample/subtask2-homographic-test.xml')
    root = tree.getroot()
    original_sentences = []
    sentences = []
    text_ids= []
    vocab = model.vocab.keys()

    for child in root:
        original_sentence = []
        text_id = child.attrib['id']
        for i in range(len(child)):
            original_sentence.append(child[i].text.lower())
        original_sentences.append(original_sentence)
        text_ids.append(text_id)

    for child in root:
        sentence = []
        for i in range(len(child)):
            if child[i].text.lower() not in stop_words and child[i].text.lower() in vocab:
                sentence.append(child[i].text.lower())
        sentences.append(sentence)

    print(text_ids[0])
    print(original_sentences[0])
    print(sentences[0])
    print(text_ids[2])
    print(original_sentences[2])
    print(sentences[2])

    with open("task2_DungLe.txt", "w") as file:
        for i in range(len(sentences)):
            result = text_ids[i] + " " + text_ids[i] + "_" + str(get_pun_word(original_sentences[i], sentences[i])) + "\n"
            file.write(result)
