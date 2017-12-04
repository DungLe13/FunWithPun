#!/usr/bin/env python3
"""
    senses_assembler.py - Task 3: Pun Interpretation assembles 2 senses for scoring
    Author: Dung Le (dungle@bennington.edu)
    Date: 12/03/2017
"""

import xml.etree.ElementTree as ET

with open('predictions/task3_sense1.txt', 'r') as sense1_file:
    senses1 = [line.strip() for line in sense1_file]
sense1_file.close()

with open('predictions/task3_sense2.txt', 'r') as sense2_file:
    senses2 = [line.strip() for line in sense2_file]
sense2_file.close()

# Load dataset from xml file (task 3)
tree3 = ET.parse('../sample/subtask3-homographic-test.xml')
root3 = tree3.getroot()
pun_ids= []

for child in root3:
    for i in range(len(child)):
        if child[i].attrib['senses'] == "2":
            pun_ids.append(child[i].attrib['id'])

with open('predictions/task3_predictions.txt', 'w') as final_pred:
    for i in range(len(pun_ids)):
        if senses1[i] == 'none' or senses2[i] == 'none':
            continue
        final_pred.write(pun_ids[i] + ' ' + senses1[i] + ' ' + senses2[i] + '\n')
final_pred.close()
