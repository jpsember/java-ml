#!/usr/bin/env python3

from pycore.base import *

import torch
import torch.nn as nn



input = torch.rand(3, 5)

# create a tensor of shape [3] with uninitialized data
#
x = torch.empty(3, dtype = torch.long)

# Fill tensor with random values 0..4
y = x.random_(5)
pr("y:")
pr(y)


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
