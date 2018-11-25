#! /usr/bin/python3

from pprint import pprint
import random
import operator
import time
from collections import defaultdict

# to format output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def remove_punctuation(line):
    line = line.replace("\n", "")
    line = line.replace(",", "")
    line = line.replace(";", "")
    line = line.replace(":", "")
    # add end of sentence marker
    line = line.replace("?", " </s>")
    line = line.replace("!", " </s>")
    line = line.replace(".", " </s>")
    return line

def preprocessing(line):
    line = remove_punctuation(line)
    wordslist = line.split(" ")
    # add start of sentence marker
    wordslist.insert(0,'<s>')
    wordslist = list(map(lambda word: word.lower(), wordslist))
    return wordslist

def unigram_frequencies(wordslist, length):
    freq_hash = {}
    for i in range (0, length):
        freq_hash[wordslist[i]] = 0
    for i in range (0, length):
        freq_hash[wordslist[i]] += 1
    # create again a sorted list
    frequencies = []
    for i in range(0, length):
        frequencies.append(freq_hash.get(wordslist[i]))

    return frequencies

# takes a probs_dict {w_i:p(w_i)} and samples a word
def unigram_sampler(probs_dict):
    # order probabilities in descending order
    probs = sorted(probs_dict.items(), key = operator.itemgetter(1), reverse = True)
    # generate a random number x in (0,1)
    x = random.uniform(0,1)
    sum = 0
    for i in range (0, len(probs)):
        sum = sum + probs[i][1] # the second value of the tuple is the value of the dict
        if sum - x >= 0:
            w_j = probs[i][0] # get the key, or the word
            return w_j

    return '</s>'

def unigram_generator(probs_dict):
    eos = False
    i = 0
    w_list = []

    while not eos:
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
            if '<s>' in probs_dict[k]:
                    probs[k] = probs_dict[k]['<s>']

        w_j = unigram_sampler(probs)
        return w_j
    else:
        probs = {}
        for k, v in probs_dict.items():
            if given_word in probs_dict[k]:
                    probs[k] = probs_dict[k][given_word]
        w_j = unigram_sampler(probs)
        return w_j

    return '</s>'

def bigram_generator(probs_dict):
    eos = False
    i = 0
    w_list = []

    while not eos:
        if i == 0:
            w_j = bigram_sampler(probs_dict, given_word = '<s>')
            w_list.append(w_j)
        else:
            w_j = bigram_sampler(probs_dict, given_word = w_list[i-1])
            if w_j != '<s>':
                w_list.append(w_j)

            if w_j == "</s>":
                eos = True
        i = i + 1

    return w_list

def trigram_sampler(probs_dict, bigram_probs_dict, first_given_word, second_given_word):
    if second_given_word == None:
        w_j = bigram_sampler(bigram_probs_dict, given_word=first_given_word)
        return w_j
    else: # both given words are != None
        # generate w_i given wi_1 and w_i-2
        probs = {}
        for k, v in probs_dict.items():
            if first_given_word in probs_dict[k] and second_given_word in probs_dict[k][first_given_word]:
                    probs[k] = probs_dict[k][first_given_word][second_given_word]
        w_j = unigram_sampler(probs)
        return w_j

    return '</s>'

def trigram_generator(probs_dict, bigram_probs_dict):
    eos = False
    i = 0
    w_list = []
    while not eos:
        if i == 0:
            w_j = trigram_sampler(probs_dict, bigram_probs_dict, first_given_word='<s>', second_given_word=None)
        elif i == 1:
            w_j = trigram_sampler(probs_dict, bigram_probs_dict, first_given_word=w_list[i-1], second_given_word=None)
        else:
            w_j = trigram_sampler(probs_dict, bigram_probs_dict, first_given_word=w_list[i-1], second_given_word=w_list[i-2])

        if w_j != '<s>':
            w_list.append(w_j)

        if w_j == "</s>":
            eos = True

        i = i + 1

    return w_list
