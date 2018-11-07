#! /usr/bin/python3

from ngram_models import *

def main():
    words = {}
    wordslist = []
    corpus = open("corpus.txt", "r")
    i = 0
    for line in corpus.readlines():
        wordslist += preprocessing(line)
        if i >= 2: # for testing purposes
            break
        i += 1

    #wordslist = ['ciao', 'a', 'tutti', 'amici', 'ciao', 'a', 'voi', 'tutti', '.']
    print('wordslist : ', wordslist, '\n')
    length = len(wordslist) # corpus length

    ##### UNIGRAM #####
    frequencies = unigram_frequencies(wordslist=wordslist, length=length)
    #print(frequencies)
    # unigram dict is made of pair key/values where key = w_i and val = f(w_i)
    unigram_dict = dict(zip(wordslist, frequencies))


    ##### BIGRAM #####
    # w_i : { w_i-1 : f(w_i, w_i-1)}
    bigram_dict = dict.fromkeys(wordslist)
    for k in bigram_dict:
        bigram_dict[k] = {}
    # init frequency of the first word
    #bigram_dict[wordslist[0]][None] = unigram_dict[wordslist[0]]
    bigram_dict[wordslist[0]]['<s>'] = unigram_dict[wordslist[0]] # not sure TODO

    for w_i in unigram_dict:
        for j in range(1, length):
            if wordslist[j] == w_i:
                if wordslist[j-1] in bigram_dict[w_i]:
                    # increase frequency
                    bigram_dict[w_i][wordslist[j-1]] += 1
                else:
                    bigram_dict[w_i][wordslist[j-1]] = 1

    print('bigram_dict : ')
    pprint(bigram_dict)

    ##### TRIGRAM #####
    # w_i : { wi_1 : { wi_2 : f(w_i,w_i-1,w_i-2)}}
    trigram_dict = dict.fromkeys(wordslist)
    for k1 in trigram_dict:
        trigram_dict[k1] = {}

    #init first two symbols TODO
    #bigram_dict[wordslist[0]]['<s>'] = unigram_dict[wordslist[0]] # not sure TODO

    for w_i in unigram_dict:
        for j in range(2, length):
            if wordslist[j] == w_i:
                if wordslist[j-1] in trigram_dict[w_i] and wordslist[j-2] in trigram_dict[w_i][wordslist[j-1]]:
                    print(True)
                    # increase frequency
                    trigram_dict[w_i][wordslist[j-1]][wordslist[j-2]] += 1
                else:
                    trigram_dict[w_i][wordslist[j-1]]= {} # create the key first
                    trigram_dict[w_i][wordslist[j-1]][wordslist[j-2]] = 1

    #print('trigram_dict : ')
    #pprint(trigram_dict)

    ##### COMPUTE NGRAM PROBABILITIES #####

    trigram_prob = dict(trigram_dict)
    for k1, v1 in trigram_prob.items():
        for k2, v2 in v1.items():
            for k3 in v2:
                # replace frequencies with probabilities
                trigram_prob[k1][k2][k3] = trigram_prob[k1][k2][k3] / float(bigram_dict[k2][k3])

    # w_i : { w_i-1 : p(w_i | w_i-1)} with probabilities instead of frequencies
    bigram_prob = dict(bigram_dict) # this is not a hard copy
    for k1, v1 in bigram_prob.items():
        for k in v1:
            bigram_prob[k1][k] = bigram_prob[k1][k] / float(unigram_dict[k])

    # dict {w_i : p(w_i)}
    unigram_prob = dict(zip(wordslist, list(map(lambda freq: float(freq)/length, frequencies))))


    unigram_sentence = unigram_generator(None, probs_dict = unigram_prob)
    #print('the sentence generated with unigram is : ', unigram_sentence)
    #index = bigram_sampler(bigram_prob, '<s>')
    #index2 = bigram_sampler(bigram_prob, 'your')
    #print('index :', index, 'index2', index2)

    ##### CONTROL PRINTS #####
    print('unigram_dict :')
    pprint(unigram_dict)
    print('bigram_dict : ')
    pprint(bigram_dict)
    print('trigram_dict : ')
    pprint(trigram_dict)
    print('unigram_prob :')
    pprint(unigram_prob)
    print('bigram_prob : ')
    pprint(bigram_prob)
    print('trigram_prob : ')
    pprint(trigram_prob)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
