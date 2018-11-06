#! /usr/bin/python3

from pprint import pprint
import random
import operator
import time

def remove_punctuation(line):
    line = line.replace("\n", "")
    line = line.replace(",", "")
    line = line.replace(";", "")
    line = line.replace(":", "")
    line = line.replace("?", " ?")
    line = line.replace("!", " !")
    line = line.replace(".", " .")
    return line

def preprocessing(line):
    line = remove_punctuation(line)
    wordslist = line.split(" ")
    # add start and end of sentence markers
    wordslist.insert(0,'<s>')
    wordslist.append('</s>')
    wordslist = list(map(lambda word: word.lower(), wordslist))

    return wordslist

def unigram_frequencies(wordslist, length):
    freq_hash = {}
    for i in range (0, length):
        freq_hash[wordslist[i]] = 0
    for i in range (0, length):
        freq_hash[wordslist[i]] += 1
    #print('freq_hash :', freq_hash)
    # create again a sorted list
    frequencies = []
    for i in range(0, length):
        frequencies.append(freq_hash.get(wordslist[i]))

    return frequencies

# for the moment works only with unigram model TODO
# takes a probs_dict (w_i:p(w_i) and samples a word)
def unigram_sampler(probs_dict):
    # order probabilities in descending order
    # probs.sort(reverse = True)
    probs = sorted(probs_dict.items(), key = operator.itemgetter(1), reverse = True)
    #print(probs)
    # generate a random number x in (0,1)
    x = random.uniform(0,1)
    for j in range(0, len(probs)): # da 0 a k-1
        sum = 0
        for i in range (0, j):
            sum = sum + probs[i][1] # the second value of the tuple is the value of the dict
        #print("sum - x : ", sum - x)
        if sum - x >= 0:
            w_j = probs[i][0] # get the key, or the word
            # j is the outcome index
            # print(j)
            return w_j

    return

# model can be 'unigram', 'bigram' or 'trigram'
#todo you can't generate a <s>
def unigram_generator(model, probs_dict):
    eos = False
    i = 0
    w_list = []

    while not eos:
        '''
        if model == 'unigram':
        elif model == 'bigram':
        else:
        '''
        w_j = unigram_sampler(probs_dict)
        if w_j != '<s>':
            w_list.append(w_j)
        if w_j == "</s>":
            eos = True
        i = i + 1

    return w_list
# probs_dict contains all the conditional probabilities, given_word could be w_i or <s>
def bigram_sampler(probs_dict, given_word):

    if given_word == '<s>':
        probs = {}
        for k, v in probs_dict.items():
            print(k,v)
            # se c'è s come inner key
            if '<s>' in probs_dict[k]:
                    probs[k] = probs_dict[k]['<s>']
        w_j = unigram_sampler(probs)
        return w_j
    else:
        probs = {}
        for k, v in probs_dict.items():
            print(k,v)
            # se c'è s come inner key
            if given_word in probs_dict[k]:
                    probs[k] = probs_dict[k][given_word]
        w_j = unigram_sampler(probs)
        print(w_j)
        return w_j
        '''
        probs = sorted(probs_dict.items(), key = operator.itemgetter(1), reverse = True)
        x = random.uniform(0,1)
        for j in range(0, len(probs)):
            sum = 0
            for i in range (0, j):
                sum = sum + probs[i][1]
            print("sum - x : ", sum - x)
            if sum - x >= 0:
                w_j = probs[i][0]
                print(j)
                return w_j
        '''
    return

def bigram_generator(probs_dict):
    eos = False
    i = 0
    w_list = []

    while not eos:

        if i == 0:
            unigram_sampler(probs_dict) #TODO finish
        else:
            w_j = unigram_sampler(probs_dict)
            if w_j != '<s>':
                w_list.append(w_j)

        if w_j == "</s>":
            eos = True
        i = i + 1

    return w_list
