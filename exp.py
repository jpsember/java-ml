#!/usr/bin/env python3

from pycore.base import *

import torch
import torch.nn as nn






a = torch.randn(3, 2, 5)
pr("a:",a)

b = a.view(-1,5)
pr("b:",b)

c = b + 5
pr("c:",c)

d = c.view(3,2,5)
pr("d:",d)
exit(0)









src  = torch.rand(3,1,1)
loss = nn.CrossEntropyLoss()
target = torch.empty(3, dtype = torch.long).random_(5)

pr("src:",src)
pr("target:",target)

output = loss(src, target)

pr("loss:",output)
exit(0)

# Construct 3 images, each with a vector of 4 class probabilities and some other dimensions
input = torch.randn(3,4)
pr("input:",input)


exit(0)





target = torch.empty(3, dtype = torch.long).random_(5)


pr("input:", input)
pr("target:",target)

print('input:\n ', input)
print('target:\n ', target)
print('Cross Entropy Loss: \n', output)







t = torch.rand(8,1,1,3)
b = t.view(8,3)
pr("t:",t)
pr("b:",b)
exit(0)


a = torch.randn(3,1,4)
pr("a:")
pr(a)
b = torch.argmax(a, dim=2, keepdim=True)
pr("argmax:")
pr(b)
exit(0)





target = torch.empty(3, dtype = torch.long).random_(5)

# Construct a 'function'?  Is this a module?
#

loss = nn.CrossEntropyLoss()
output = loss(input, target)

pr("input:", input)
pr("target:",target)

print('input:\n ', input)
print('target:\n ', target)
print('Cross Entropy Loss: \n', output)
