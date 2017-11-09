#!/usr/bin/env python3
"""
    task1_baseline.py - Task 1: Pun Detection choosing random word
    Author: Dung Le (dungle@bennington.edu)
    Date: 11/07/2017
"""

import xml.etree.ElementTree as ET
import random

def classify_random_pun_sentence():
    with open("baselines/task1_baseline_random.txt", "w") as file:
        for child in root:
            text_id = child.attrib['id']
            rand_pun_number = random.randint(0, 1)
            pun_prediction = text_id + " " + str(rand_pun_number) + "\n"
            file.write(pun_prediction)
    file.close()

if __name__ == "__main__":
    # Load dataset from xml file (task 1)
    tree = ET.parse('../sample/subtask1-homographic-test.xml')
    root = tree.getroot()

    classify_random_pun_sentence()
