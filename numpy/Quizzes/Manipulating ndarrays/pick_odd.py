# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:34:11 2020

@author: dilayerc
"""


import numpy as np

# Create a 5 x 5 ndarray with consecutive integers from 1 to 25 (inclusive).
# Afterwards use Boolean indexing to pick out only the odd numbers in the array

# Create a 5 x 5 ndarray with consecutive integers from 1 to 25 (inclusive).
X =  np.arange(1,26).reshape(5,5)

# Use Boolean indexing to pick out only the odd numbers in the array
Y = X[X % 2 == 1]
