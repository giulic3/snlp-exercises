#!/usr/bin/python3

from pprint import pprint
import random

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
    wordslist = list(map(lambda word: word.lower(), wordslist))
    return wordslist

# probs is a list of probabilities
def draw_samples(probs):
    # order probabilities in descending order
    probs.sort(reverse = True)
    print(probs)
    # generate a random number x in (0,1)
    x = random.uniform(0,1)

    for j in range(0, len(probs)): # da 0 a k-1
        sum = 0
        for i in range (0, j):
            sum = sum + probs[i]
        print("sum - x : ", sum - x)
        if sum - x >= 0:
            print('in if!')
            print('return sel_j')
            return j

    return

def unigram_model():
    return

def bigram_model():
    return

def trigram_model():
    return


def main():
    words = {}
    wordslist = []
    corpus = open("corpus.txt", "r")
    i = 0
    for line in corpus.readlines():
        wordslist += preprocessing(line)
        if i >= 1000: # for testing purposes
            break
        i += 1

    #test:
    #wordslist = ['ciao', 'a', 'tutti', 'amici', 'ciao', 'a', 'voi', 'tutti', '.']
    #wordslist.insert(0,"<s>") # model the beginning of a sentence
    #wordslist.append("</s>") # model the end of a sentence
    print('wordslist : ')
    print(wordslist)
    print('\n')
    # create a list of frequencies of each word in the corpus
    frequencies = list(map(lambda word: wordslist.count(word), wordslist)) #TODO comp. expensive
    #print(frequencies)
    length = len(wordslist) # corpus length
    # unigram dict is made of pair key/values where key = w_i and val = f(w_i) - repeated values are discarded
    unigram_dict = dict(zip(wordslist, frequencies))
    #print(unigram_dict)
    # dict {w_i : p(w_i)}
    unigram_prob = dict(zip(wordslist, list(map(lambda freq: float(freq)/length, frequencies))))
    #print('unigram probability distribution : ', unigram_prob)
    # w_i : { w_i-1 : f(w_i | w_i-1)}
    # initialize dict with w_i as outer key
    digram_dict = dict.fromkeys(wordslist)
    for k in digram_dict:
        digram_dict[k] = {}
    # init frequency of the first word
    digram_dict[wordslist[0]][None] = unigram_dict[wordslist[0]]

    for w_i in unigram_dict:
        for j in range(1, length):
            if wordslist[j] == w_i:
                if wordslist[j-1] in digram_dict[w_i]:
                    # increase frequency
                    digram_dict[w_i][wordslist[j-1]] += 1
                else:
                    digram_dict[w_i][wordslist[j-1]] = 1
        #print('w_i : ', w_i,' digram_dict[w_i] : ' ,digram_dict[w_i])
    print('digram_dict : ')
    pprint(digram_dict)

    #digram_prob = digram_dict[i].value / freq di k TODO finish

    trigram_dict = dict.fromkeys(wordslist)
    for k1 in trigram_dict:
        trigram_dict[k1] = {}

    #pprint(trigram_dict)
    '''
    index = draw_samples(list(unigram_prob.values()))
    print('index sampled : ', index)
    '''
if __name__ == "__main__":
    main()
