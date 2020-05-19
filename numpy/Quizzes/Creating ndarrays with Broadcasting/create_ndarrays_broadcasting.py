# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:36:22 2020

@author: dilayerc
"""


import numpy as np

# Use Broadcasting to create a 4 x 4 ndarray that has its first
# column full of 1s, its second column full of 2s, its third
# column full of 3s, etc.. 

first = np.array([1, 2, 3, 4])

a = np.vstack((first, first))

b = np.vstack((a, first))

X = np.vstack((b, first))
