import pandas as pd
import numpy as np


def find_subsequence(sentence, subsentence, pos_tags):
    """ 
    Get POS tags for arg1/relation/arg2 in a best-effort way: Find whole subsequence. If not found, use LCS
    """
    for i in range(len(sentence) - len(subsentence)):
        if sentence[i:i+len(subsentence)] == subsentence:
            return pos_tags[i:i+len(subsentence)]
    return lcs(sentence, subsentence, pos_tags)

def lcs(sentence, subsentence, pos_tags):
    """
    Longest common subsequence 
    """
    s1, s2 = sentence, subsentence
    matrix = [[list() for x in range(len(s2))] for x in range(len(s1))]
    pos_matrix = [[list() for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = [s1[i]]
                    pos_matrix[i][j] = [pos_tags[i]]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + [s1[i]]
                    pos_matrix[i][j] = pos_matrix[i-1][j-1] + [pos_tags[i]]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
                pos_matrix[i][j] = max(pos_matrix[i-1][j], pos_matrix[i][j-1], key=len)

    return pos_matrix[-1][-1]
