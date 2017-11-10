#!/usr/bin/env python3
"""
    task3.py - Task 3: Pun Interpretation using word2vec, cosine similarities,
               and Lesk algorithm the compare the glosses of two words
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/09/2017
"""

import xml.etree.ElementTree as ET
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tag import StanfordPOSTagger

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
            tagged_word = pos_tagger.tag(words)
    
            if child[i].attrib['senses'] == "2":
                pun_pos.append(tagged_word[0][1])

    return pun_pos

def get_most_probable_word(sent, pun_word):
    scores = {}
    for i in range(len(sent)):
        if sent[i] != pun_word:
            sim_score = model.similarity(sent[i], pun_word)
            scores['{0}-{1}'.format(sent[i], pun_word)] = sim_score

    for pair, score in scores.items():
        if score == max(scores.values()):
            return pair.split(sep='-')[0]

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
        pun_pos = pos_tagger.tag([pun_word])[0][1]
        if pun_pos == 'NN' or pun_pos == 'NNS' or pun_pos == 'NNP':
            pun_pos = 'n'
        elif pun_pos == 'VB' or pun_pos == 'VBD' or pun_pos == 'VBG' or pun_pos == 'VBN' or pun_pos == 'VBP' or pun_pos == 'VBZ':
            pun_pos = 'v'
        elif pun_pos == 'JJ' or pun_pos == 'JJR' or pun_pos == 'JJS':
            pun_pos = 's'

    return pun_word, pun_pos

def most_common_sense(pun_word, pun_pos):
    pun_word, pun_pos = correct_pun_pos(pun_word, pun_pos)
    print(pun_word)
    print(pun_pos)

    if wn.synsets(pun_word, pos=pun_pos):
        print(len(wn.synsets(pun_word, pos=pun_pos)))
        most_common_synset = wn.synsets(pun_word, pos=pun_pos)[0]
        most_common_sense = most_common_synset.lemmas()[0].key()
        return most_common_sense
    else:
        # if the list of synsets is empty, try to stem the word, get a new POS, and find synsets of the new pair
        pun_word = porter_stemmer.stem(pun_word)
        pun_pos = pos_tagger.tag([pun_word])[0][1]
        pun_word, pun_pos = correct_pun_pos(pun_word, pun_pos)
        print(pun_word)
        print(pun_pos)

        most_common_synset = wn.synsets(pun_word, pos=pun_pos)[0]
        most_common_sense = most_common_synset.lemmas()[0].key()
        return most_common_sense

if __name__ == "__main__":
    # Load Google's pre-trained Word2Vec model.
    # model = gensim.models.KeyedVectors.load_word2vec_format('../sample/GoogleNews-vectors-negative300.bin', binary=True)

    # Paths for Stanford POS Tagger
    jar = '../../stanford-postagger-full-2016-10-31/stanford-postagger.jar'
    model = '../../stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf-8')
    
    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')

    # Load dataset from xml file (task 2)
    tree = ET.parse('../sample/subtask3-homographic-test.xml')
    root = tree.getroot()
    sentences = []
    text_ids= []
    # vocab = model.vocab.keys()
    porter_stemmer = PorterStemmer()

    '''
    for child in root:
        sentence = []
        for i in range(len(child)):
            if child[i].text.lower() not in stop_words and child[i].text.lower() in vocab:
                sentence.append(child[i].text.lower())
        sentences.append(sentence)
    '''

    pun_words = get_pun_word()
    pun_pos = pos_for_pun_word()

    for i in range(len(sentences)):
        #get_most_probable_word(sentences[i], pun_words[i])
        #pun_pos[i]
        print(most_common_sense(pun_words[i], pun_pos[i]))
    
    # print(get_most_probable_word(sentences[10], pun_words[10]))
    # print(pun_pos[10])
    print(most_common_sense(pun_words[10], pun_pos[10]))
