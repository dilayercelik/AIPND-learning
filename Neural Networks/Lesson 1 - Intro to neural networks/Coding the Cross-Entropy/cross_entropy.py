# Coding the Cross-Entropy

## see formula in image file


import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

## Y is for the categories and P is for the probabilities (both lists)

def cross_entropy(Y, P):

    CE = 0

    for i in range(len(Y)):
        CE -= Y[i] * np.log(P[i]) + (1 - Y[i]) * np.log(1 - P[i])

    return CE
