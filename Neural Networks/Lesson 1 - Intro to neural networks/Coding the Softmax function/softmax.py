# Coding the Softmax function

## used to compute the probability of each class
## when the classification problem involves more than 2 classes
## when 2 classes: if discrete => step function / if continuous => sigmoid function


import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):

    results_list = []

    denominator_sum = 0
    for i in range(len(L)):
        denominator_sum += np.exp(L[i])


    for score in L:
        result_score = np.exp(score) / denominator_sum

        results_list.append(result_score)

    return results_list
