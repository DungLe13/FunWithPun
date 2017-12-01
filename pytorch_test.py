import torch
import torch.nn as nn
from torch.autograd import Variable

time_steps = 15
batch_size = 3
embeddings_size = 100
num_classes = 2

model = nn.LSTM(embeddings_size, num_classes)
input_seq = Variable(torch.randn(time_steps, batch_size, embeddings_size))
lstm_out, _ = model(input_seq)
last_out = lstm_out[-1]
print(last_out)

loss = nn.BCELoss()
target = Variable(torch.LongTensor(batch_size).random_(0, num_classes))
print(target)
err = loss(torch.nn.functional.softmax(last_out)[:,1], target.float())
print(err)
err.backward()

'''
score = Variable(torch.randn(10,2))
print(score)
target = Variable((torch.rand(10)>0.5).long())
print(target)
lfn1 = torch.nn.CrossEntropyLoss()
lfn2 = torch.nn.BCELoss()
print(lfn1(score,target),
      lfn2(torch.nn.functional.softmax(score)[:,1],target.float()))
'''
