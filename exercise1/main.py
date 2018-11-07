#! /usr/bin/python3

from ngram_models import *

def main():
    words = {}
    wordslist = []
    corpus = open("corpus.txt", "r")
    i = 0
    for line in corpus.readlines():
        wordslist += preprocessing(line)
        i += 1

    length = len(wordslist) # corpus length

    ##### UNIGRAM #####
    frequencies = unigram_frequencies(wordslist=wordslist, length=length)
    # { w_i : f(w_i) }
    unigram_dict = dict(zip(wordslist, frequencies))

    ##### BIGRAM #####
    # { w_i : { w_i-1 : f(w_i, w_i-1)} }
    bigram_dict = dict.fromkeys(wordslist)
    for k in bigram_dict:
        bigram_dict[k] = {}

    for j in range(1, length):
        if wordslist[j-1] in bigram_dict[wordslist[j]]:
            # increase frequency
            bigram_dict[wordslist[j]][wordslist[j-1]] += 1
        else:
            bigram_dict[wordslist[j]][wordslist[j-1]] = 1


    ##### TRIGRAM #####
    # { w_i : { wi_1 : { wi_2 : f(w_i,w_i-1,w_i-2)}} }
    trigram_dict = dict.fromkeys(wordslist)
    for k1 in trigram_dict:
        trigram_dict[k1] = {}

    for j in range(2, length):
        if wordslist[j-1] in trigram_dict[wordslist[j]] and wordslist[j-2] in trigram_dict[wordslist[j]][wordslist[j-1]]:
            # increase frequency
            trigram_dict[wordslist[j]][wordslist[j-1]][wordslist[j-2]] += 1
        else:
            trigram_dict[wordslist[j]][wordslist[j-1]]= {} # create the key first
            trigram_dict[wordslist[j]][wordslist[j-1]][wordslist[j-2]] = 1

    ##### COMPUTE NGRAM PROBABILITIES #####
    trigram_prob = dict(trigram_dict)
    for k1, v1 in trigram_prob.items():
        for k2, v2 in v1.items():
            for k3 in v2:
                # replace frequencies with probabilities
                trigram_prob[k1][k2][k3] = trigram_prob[k1][k2][k3] / float(bigram_dict[k2][k3])

    # { w_i : { w_i-1 : p(w_i | w_i-1)} } with probabilities instead of frequencies
    bigram_prob = dict(bigram_dict) # this is not a hard copy
    for k1, v1 in bigram_prob.items():
        for k in v1:
            bigram_prob[k1][k] = bigram_prob[k1][k] / float(unigram_dict[k])

    # dict {w_i : p(w_i)}
    unigram_prob = dict(zip(wordslist, list(map(lambda freq: float(freq)/length, frequencies))))

    unigram_sentence = unigram_generator(probs_dict=unigram_prob)
    bigram_sentence = bigram_generator(probs_dict=bigram_prob)
    trigram_sentence = trigram_generator(probs_dict=trigram_prob, bigram_probs_dict=bigram_prob)

    print(bcolors.OKBLUE + 'the sentence generated with unigram is : ' + bcolors.ENDC, unigram_sentence)
    print(bcolors.OKBLUE + 'the sentence generated with bigram is : ' + bcolors.ENDC, bigram_sentence)
    print(bcolors.OKBLUE + 'the sentence generated with trigram is : '  + bcolors.ENDC, trigram_sentence)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
