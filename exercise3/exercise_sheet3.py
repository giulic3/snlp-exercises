################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import numpy as np
import pprint


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
        if not line:
            break

        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))

    f.close()
    return sentences


class MaxEntModel(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None
    
    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    # { (word:label) : index }
    feature_indices = None
    
    # set containing a list of possible labels
    # has to be set by the method 'initialize'
    labels = None

    def __init__(self):

        self.corpus = import_corpus('./corpus_fake.txt')
        self.feature_indices = {}
        self.labels = set()
        self.initialize(self.corpus)

    # Exercise 1 a) ###################################################################
    '''
    Initialize the maximum entropy model, i.e., build the set of all features, the set of all labels
    and create an initial array 'theta' for the parameters of the model.
    Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
    '''

    def initialize(self, corpus):
        # (the, DT), (dog, NN)
        self.corpus = corpus
        # set of all words
        X = set()
        # set of features as tuples
        F = set()
        F_list = []

        # fill the sets with elements
        self.labels.add('start')
        for sentence in corpus:
            for pair in sentence:
                X.add(pair[0])
                self.labels.add(pair[1])

        # build the set F of all features
        for word in X:
            for label in self.labels:
                F.add((word, label))
        for first_label in self.labels:
            for second_label in self.labels:
                F.add((first_label, second_label))

        for feature in F:
            F_list.append(feature)
        # store features and their indices into a dictionary
        for index in range(0, len(F_list)):
            self.feature_indices[F_list[index]] = index

        # initialize the vector of parameters as np array filled with 1
        self.theta = np.ones((len(F_list),), dtype=int)

        # initialize the vector of parameters
        # self.theta = [1 for feature in self.feature_indices]

        print(Colors.OKBLUE + "F_list: " + Colors.ENDC, F_list)
        print(Colors.OKBLUE + "corpus: " + Colors.ENDC, self.corpus)
        print(Colors.OKBLUE + "feature_indices: " + Colors.ENDC, self.feature_indices)
        print(Colors.OKBLUE + "theta: " + Colors.ENDC, self.theta)

    # Exercise 1 b) ###################################################################
    '''
    Compute the vector of active features.
    Parameters: word: string; a word at some position i of a given sentence
                label: string; a label assigned to the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing only zeros and ones.
    '''

    def get_active_features(self, word, label, prev_label):

        # init array of active features
        active_features = np.zeros(len(self.feature_indices))

        for feature in self.feature_indices:
            if feature == (word, label):
                active_features[self.feature_indices[(word, label)]] = 1
            elif feature == (prev_label, label):
                active_features[self.feature_indices[(prev_label, label)]] = 1

        # print(Colors.OKBLUE + "active_features: " + Colors.ENDC, active_features)

        return active_features
    # Exercise 2 a) ###################################################################
    '''
    Compute the normalization factor 1/Z(x_i).
    Parameters: word: string; a word x_i at some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: float
    '''
    # TODO use numpy functions everywhere!
    # np.prod([[1.,2.],[3.,4.]], axis=1) dà [2. , 12.]
    def cond_normalization_factor(self, word, prev_label):
        z = 0
        for label in self.labels:
            x = 0
            active_features = self.get_active_features(word, label, prev_label)
            for j in range(len(self.feature_indices)):
                x += self.theta[j]*active_features[j]
            z += math.exp(x)

        print(Colors.OKBLUE + "1/Z(" + word + "): " + Colors.ENDC, np.reciprocal(z))
        return np.reciprocal(z)

    # Exercise 2 b) ###################################################################
    '''
     Compute the conditional probability of a label given a word x_i.
     Parameters: label: string; we are interested in the conditional probability of this label
                 word: string; a word x_i some position i of a given sentence
                 prev_label: string; the label of the word at position i-1
     Returns: float
     '''

    def conditional_probability(self, label, word, prev_label):

        # TODO
        # count occurrences of the label
        # count occurrences of the label associated to the word x_i
        pass

    # Exercise 3 a) ###################################################################
    '''
    Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
    Parameters: word: string; a word x_i some position i of a given sentence
                label: string; the actual label of the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the empirical feature count
    '''

    def empirical_feature_count(self, word, label, prev_label):

        # your code here
        
        pass

    # Exercise 3 b) ###################################################################
    '''
    Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
    (see variable theta)
    Parameters: word: string; a word x_i some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the expected feature count
    '''
    def expected_feature_count(self, word, prev_label):

        # your code here
        
        pass
    
    # Exercise 4 a) ###################################################################
    '''
    Do one learning step.
    Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                label: string; the actual label of the selected word
                prev_label: string; the label of the word at position i-1
                learning_rate: float
    '''

    def parameter_update(self, word, label, prev_label, learning_rate):

        # your code here
        
        pass

    # Exercise 4 b) ###################################################################
    '''
    Implement the training procedure.
    Parameters: number_iterations: int; number of parameter updates to do
                learning_rate: float
    '''

    def train(self, number_iterations, learning_rate=0.1):

        # your code here
        
        pass
    
    # Exercise 4 c) ###################################################################
    '''
     Predict the most probable label of the word referenced by 'word'
     Parameters: word: string; a word x_i at some position i of a given sentence
                 prev_label: string; the label of the word at position i-1
     Returns: string; most probable label
     '''
    def predict(self, word, prev_label):
        
        # your code here
        
        pass
    

