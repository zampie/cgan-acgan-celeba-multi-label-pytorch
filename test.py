import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
device = torch.device("cpu")

l1 = torch.tensor([[2.,3.,4.],[-1.,2.,3.]],device=device)

si = torch.sigmoid(l1)

so0 = torch.softmax(l1,0)
so1 = torch.softmax(l1,1)