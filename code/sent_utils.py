import json
import numpy as np
import lemminflect 

def add_fullstop(sent):
    if sent.endswith('.'):
        return sent
    else:
        return sent+'.'

def remove_fullstop(sent):
    if sent.endswith('.'):
        return sent[:-1]
    else:
        return sent


def LCstring(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                result = max(result,res[i][j])  
    return result


def sent_overlap(sent1,sent2,spacy_nlp,thre=-1):
    
    spacy_nlp.Defaults.stop_words -= {"using", "show","become","make","down","made","across","put","see","move","part","used"}
    
    doc1 = spacy_nlp(sent1)
    doc2 = spacy_nlp(sent2)    
    
    word_set1 = set([token.lemma_ for token in doc1 if not (token.is_stop or token.is_punct)])
    word_set2 = set([token.lemma_ for token in doc2 if not (token.is_stop or token.is_punct)])
    
    if thre == -1:
        # do not use LCstring
        if len(word_set1.intersection(word_set2)) > 0:
            return True
        else:
            return False

    # use LCstring
    max_socre = -1
    for word1 in word_set1:
        for word2 in word_set2:
            lcs = LCstring(word1,word2)
            score = lcs / min(len(word1),len(word2))
            max_socre = score if score > max_socre else max_socre
            
    if max_socre > thre:
        return True
    else:
        return False
             