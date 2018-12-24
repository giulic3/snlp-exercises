################################################################################
# SNLP exercise sheet 4
################################################################################
import math
import sys
import numpy as np
import time


# Class used to format output and improve readability
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''


def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)

    while True:
        line = f.readline()
        if not line: break

        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))

    f.close()
    return sentences


class LinearChainCRF(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None

    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    # { (word:label) : index }
    feature_indices = None # TODO do I need it?

    # set containing all features observed in the corpus 'self.corpus'
    # choose an appropriate data structure for representing features
    # each element of this set has to be assigned to exactly one component of the vector 'self.theta'
    features = None
    
    # set containing all labels observed in the corpus 'self.corpus'
    labels = None

    def initialize(self, corpus):

        F = set()

        self.corpus = corpus

        self.feature_indices = {}
        # build set self.labels
        self.labels.add('start')
        F.add(('start', corpus[0][1]))
        F.add((corpus[0][0], corpus[0][1]))

        for sentence in corpus:
            for i in range(1, len(sentence)):
                word = sentence[i][0]
                label = sentence[i][1]
                prev_label = sentence[i-1][1]
                self.labels.add(label)
                F.add((word, label))
                F.add((prev_label, label))

        # build dict of feature_indices
        index = 0
        for feature in self.features:
            self.feature_indices[feature] = index
            index += 1

        for feature in self.features:
            self.features[feature] = self.get_active_features(...) # get_active_features...


        # initialize the vector of parameters as np array filled with 1
        self.theta = np.ones(len(self.features), dtype=float)

    # Exercise 1 a) ###################################################################
    '''
    Compute the forward variables for a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: data structure containing the matrix of forward variables
    '''
    def forward_variables(self, sentence):

        # your code here
        
        pass

    '''
    Compute the backward variables for a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: data structure containing the matrix of backward variables
    '''
    def backward_variables(self, sentence):

        # your code here
        
        pass

    '''
    Compute the partition function Z(x).
    Parameters: sentence: list of strings representing a sentence.
    Returns: float;
    '''
    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence):

        # your code here
        
        pass
            
    # Exercise 1 c) ###################################################################
    '''
    Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
    Parameters: sentence: list of strings representing a sentence.
                y_t: element of the set 'self.labels'; label assigned to the word at position t
                y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
    Returns: float: probability;
    '''
    def marginal_probability(self, sentence, y_t, y_t_minus_one):

        # your code here
        
        pass
    
    # Exercise 1 d) ###################################################################
    '''
    Compute the expected feature count for the feature referenced by 'feature'
    Parameters: sentence: list of strings representing a sentence.
                feature: a feature; element of the set 'self.features'
    Returns: float;
    '''
    def expected_feature_count(self, sentence, feature):

        # your code here
        
        pass

    '''
    Method for training the CRF.
    Parameters: num_iterations: int; number of training iterations
                learning_rate: float
    '''
    # Exercise 1 e) ###################################################################
    def train(self, num_iterations, learning_rate=0.01):

        # your code here
        
        pass
    
    # Exercise 2 ###################################################################
    '''
    Compute the most likely sequence of labels for the words in a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: list of lables; each label is an element of the set 'self.labels'
    '''
    def most_likely_label_sequence(self, sentence):
        
        # your code here
        
        pass

    
def main():

    corpus = import_corpus('./corpus_pos.txt')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))