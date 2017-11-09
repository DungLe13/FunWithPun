#!/usr/bin/env python3
"""
    task2_baselines.py - Task 2: Pun Location choosing random word, last word, or word with most senses
    Author: Dung Le (dungle@bennington.edu)
    Date: 10/30/2017
"""

import xml.etree.ElementTree as ET
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

def get_random_pun_word():
    with open("baselines/task2_baseline1_random.txt", "w") as file:
        for child in root:
            text_id = child.attrib['id']
            rand_pun_number = random.randint(1, len(child)-1)
            pun_prediction = text_id + " " + text_id + "_" + str(rand_pun_number) + "\n"
            file.write(pun_prediction)
    file.close()

def get_last_word():
    with open("baselines/task2_baseline2_last.txt", "w") as file:
        for child in root:
            text_id = child.attrib['id']
            last_word_number = len(child) - 1
            pun_prediction = text_id + " " + text_id + "_" + str(last_word_number) + "\n"
            file.write(pun_prediction)
    file.close()

def get_word_with_most_senses():
    with open("baselines/task2_baseline3_mostsenses.txt", "w") as file:
        for child in root:
            senses_number = 0
            prev_word_id = ""
            text_id = child.attrib['id']
            for i in range(len(child)):
                if child[i].text.lower() not in stop_words:
                    word_senses_num = len(wn.synsets(child[i].text.lower()))
                    current_word_id = child[i].attrib['id']
                    if word_senses_num >= senses_number:
                        senses_number = word_senses_num
                        prev_word_id = current_word_id
            pun_prediction = text_id + " " + prev_word_id + "\n"
            file.write(pun_prediction)
    file.close()

if __name__ == "__main__":
    # Load dataset from xml file (task 2)
    tree = ET.parse('../sample/subtask2-homographic-test.xml')
    root = tree.getroot()

    stop_words = set(stopwords.words('english'))
    stop_words.update('.', '?', '-', '\'', '\:', ',', '!', '<', '>', '\"', '/', '(', ')',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 's', 't', 're', 'm')

    get_random_pun_word()
    get_last_word()
    get_word_with_most_senses()
