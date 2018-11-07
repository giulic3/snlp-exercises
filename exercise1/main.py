#! /usr/bin/python3

from ngram_models import *

def main():
    words = {}
    wordslist = []
    corpus = open("corpus.txt", "r")
    i = 0
    for line in corpus.readlines():
        wordslist += preprocessing(line)
        #if i >= 1000: # for testing purposes
        #    break
        i += 1

        '''
    wordslist = [
    '<s>','ciao', 'a', 'tutti', 'amici', 'ciao', 'a', 'voi', 'tutti', '.', '</s>',
    '<s>','oggi', 'sono', 'molto', 'stanca', 'sono', 'molto', 'disperata', '.','</s>',
    '<s>','ciao', 'a', 'tutti', 'ancora', '.','</s>']
    '''
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
    #bigram_dict[wordslist[0]]['<c>'] = unigram_dict[wordslist[0]] # not sure TODO
    # TODO this can be done without using the dict first
    for j in range(1, length):
        if wordslist[j-1] in bigram_dict[wordslist[j]]:
            # increase frequency
            bigram_dict[wordslist[j]][wordslist[j-1]] += 1
        else:
            bigram_dict[wordslist[j]][wordslist[j-1]] = 1


    ##### TRIGRAM #####
    # w_i : { wi_1 : { wi_2 : f(w_i,w_i-1,w_i-2)}}
    trigram_dict = dict.fromkeys(wordslist)
    for k1 in trigram_dict:
        trigram_dict[k1] = {}

    #init first two symbols TODO
    #bigram_dict[wordslist[0]]['<c>'] = unigram_dict[wordslist[0]] # not sure TODO
    # TODO devo considerare che dopo una eos non ci va nient'altro?
    for j in range(2, length):
        if wordslist[j-1] in trigram_dict[wordslist[j]] and wordslist[j-2] in trigram_dict[wordslist[j]][wordslist[j-1]]:
            # increase frequency
            trigram_dict[wordslist[j]][wordslist[j-1]][wordslist[j-2]] += 1
        else:
            trigram_dict[wordslist[j]][wordslist[j-1]]= {} # create the key first
            trigram_dict[wordslist[j]][wordslist[j-1]][wordslist[j-2]] = 1
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

    unigram_sentence = unigram_generator(probs_dict=unigram_prob)
    bigram_sample_start = bigram_sampler(probs_dict=bigram_prob, given_word='<s>')
    bigram_sample = bigram_sampler(probs_dict=bigram_prob, given_word='the')

    bigram_sentence = bigram_generator(probs_dict=bigram_prob)
    '''
    trigram_sample_start = trigram_sampler(probs_dict=trigram_prob, bigram_probs_dict=bigram_prob, first_given_word='<s>', second_given_word=None)
    trigram_sample_second = trigram_sampler(probs_dict=trigram_prob, bigram_probs_dict=bigram_prob, first_given_word='ciao', second_given_word=None)
    trigram_sample = trigram_sampler(probs_dict=trigram_prob, bigram_probs_dict=bigram_prob, first_given_word='a', second_given_word='ciao')
    '''
    trigram_sentence = trigram_generator(probs_dict=trigram_prob, bigram_probs_dict=bigram_prob)
    ##### CONTROL PRINTS #####
    #print('wordslist : ', wordslist, '\n')
    #print('unigram_dict :')
    #pprint(unigram_dict)
    #print('bigram_dict : ')
    #pprint(bigram_dict)
    #print('trigram_dict : ')
    #pprint(trigram_dict)
    #print('unigram_prob :')
    #pprint(unigram_prob)
    #print('bigram_prob : ')
    #pprint(bigram_prob)
    #print('trigram_prob : ')
    #pprint(trigram_prob)
    print(bcolors.OKBLUE + 'the sentence generated with unigram is : ' + bcolors.ENDC, unigram_sentence)
    #print('bigram_sample_start :', bigram_sample_start)
    #print('bigram_sample :', bigram_sample)
    print(bcolors.OKBLUE + 'the sentence generated with bigram is : ' + bcolors.ENDC, bigram_sentence)
    #print('trigram_sample_start : ', trigram_sample_start)
    #print('trigram_sample_second : ', trigram_sample_second)
    #print('trigram_sample : ', trigram_sample)
    print(bcolors.OKBLUE + 'the sentence generated with trigram is : '  + bcolors.ENDC, trigram_sentence)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
