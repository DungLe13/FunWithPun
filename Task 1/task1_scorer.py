#!/usr/bin/env python3
"""
    task1_scorer.py - Task 1: Pun Detection scorer using Precision, Recall, Acurracy, and F1 score
    Author: Dung Le (dungle@bennington.edu)
    Date: 12/03/2017
"""

file_name = 'train_80_test_20/hidden_300_epoch_20.txt'

with open('results/{0}'.format(file_name), 'r') as f:
    predictions = [line.strip() for line in f]
f.close()

with open('results/train_80_test_20/test_y.txt', 'r') as fr:
    results = [line.strip() for line in fr]
fr.close()

"""
    Abreviations:
    TP = True Positive
    FP = False Positive
    FN = False Negative
    TN = True Negative
"""
TP = FP = FN = TN = 0

for i in range(len(results)):
    if results[i] == '1' and predictions[i] == '1':
        TP += 1
    elif results[i] == '1' and predictions[i] == '0':
        FN += 1
    elif results[i] == '0' and predictions[i] == '1':
        FP += 1
    elif results[i] == '0' and predictions[i] == '0':
        TN += 1

print(TP, FN, FP, TN)

# PRECISION
P = TP / (TP + FP)
R = TP / (TP + FN)
A = (TP + TN) / (TP + TN + FP +FN)
F1 = 2*P*R / (P+R)

with open('scores/{0}'.format(file_name), 'w') as res:
    res.write('precision: ' + str(P) + '\n')
    res.write('recall: ' + str(R) + '\n')
    res.write('accuracy: ' + str(A) + '\n')
    res.write('f1: ' + str(F1) + '\n')
res.close()

print(P)
print(R)
print(A)
print(F1)
