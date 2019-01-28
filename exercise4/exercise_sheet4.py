################################################################################
# SNLP exercise sheet 4
################################################################################
import math
import numpy as np
import time
import random


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


class LinearChainCRF(object):
    # training corpus
    corpus = None

    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None

    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    # { (word:label) : index }
    feature_indices = None  # TODO do I need it?

    # set containing all features observed in the corpus 'self.corpus'
    # choose an appropriate data structure for representing features
    # each element of this set has to be assigned to exactly one component of the vector 'self.theta'
    features = None

    # set containing all labels observed in the corpus 'self.corpus'
    labels = None

    def initialize(self, corpus):

        self.corpus = corpus
        self.feature_indices = {}
        self.features = {}
        self.labels = []
        # set of all words
        X = set()
        # set of features as tuples
        F = set()
        labels_set = set()
        # build set self.labels
        # fill the sets with elements
        for sentence in corpus:
            for pair in sentence:
                X.add(pair[0])
                labels_set.add(pair[1])

        # build the set F of all features
        for word in X:
            for label in labels_set:
                F.add((word, label))
                F.add(('start', label))
        for first_label in labels_set:
            for second_label in labels_set:
                F.add((first_label, second_label))

        print('F: ', F)
        # build list of labels
        for label in labels_set:
            self.labels.append(label)

        print(Colors.OKBLUE + "labels: " + Colors.ENDC, self.labels)

        # build dict of feature_indices
        index = 0
        for feature in F:
            self.feature_indices[feature] = index
            index += 1

        # TODO should I init the features here? or store them only when needed?
        # for feature in F:
        #    self.features[feature] = self.get_active_features(feature)

        # initialize the vector of parameters as np array filled with 1
        self.theta = np.ones(len(self.feature_indices.keys()), dtype=float)

        print(Colors.OKBLUE + "corpus: " + Colors.ENDC, self.corpus)
        print(Colors.OKBLUE + "feature_indices: " + Colors.ENDC, self.feature_indices)
        print(Colors.OKBLUE + "theta: " + Colors.ENDC, self.theta)

    '''
    Compute the sets of active features (as theta's indexes)
    Parameters: word: string; a word at some position i of a given sentence
                label: string; a label assigned to the given word
                prev_label: string; the label of the word at position i-1
    Returns: set of active features thetas for a given tuple (label, prev_label, word)
    = (y_t, y_t-1, x_t)
    '''

    def get_active_features(self, label, prev_label, word):
        # if the active features for this key have been already computed, just return them
        if self.features.get((label, prev_label, word), None) is not None:
            return self.features[(label, prev_label, word)]
        # otherwise compute and save them
        # init set of active features param indexes (then used to retrieve theta values)
        active_thetas_indices = set()
        for feature in self.feature_indices:
            if feature == (word, label):
                active_thetas_indices.add(self.feature_indices[(word, label)])
            elif feature == (prev_label, label):
                active_thetas_indices.add(self.feature_indices[(prev_label, label)])

        self.features[(label, prev_label, word)] = active_thetas_indices

        return active_thetas_indices

    '''
    Compute the sum of thetas given correspondent feature indices.
    Parameters: Indices of theta parameters that correspond to active features.
    Returns: sum of theta values
    '''
    # TODO: see if it can be done more easily with numpy and array splitting
    def get_thetas_sum(self, active_theta_indices):

        theta_sum = 0
        for index in active_theta_indices:
            theta_sum += self.theta[index]

        return theta_sum

    '''
    Compute the factor given the correspondent active features.
    Parameters: label: string; a label assigned to the given word
                prev_label: string; the label of the word at position i-1
                word: string; a word at some position i of a given sentence
    Returns: float; the factor
    '''
    def compute_factor(self, label, prev_label, word):
        factor = math.exp(self.get_thetas_sum(self.get_active_features(label, prev_label, word)))

        return factor

    # Exercise 1 a) ###################################################################
    '''
    Compute the forward variables for a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: data structure containing the matrix of forward variables
    '''
    def forward_variables(self, sentence):
        sentence_length = len(sentence)
        # list of dictionaries: where the i_th dictionary represent the i_th word of the sentence and
        # each dict has entries of the form { label : forward_variable }
        forward_variables_matrix = []
        # init dictionaries
        for i in range(0, sentence_length):
            forward_variables_matrix.append({})
            for label in self.labels:
                forward_variables_matrix[i][label] = 0.
        # init first forward variable
        # first_label = sentence[0][1]
        first_word = sentence[0][0]
        for label in self.labels:
            forward_variables_matrix[0][label] = round(self.compute_factor(label, 'start', first_word), 2)

        for t in range(1, sentence_length):
            word = sentence[t][0]
            # prev_label = sentence[t - 1][1]
            for label in self.labels:
                summation = 0.
                for prev_label in self.labels:
                    # filling matrix per columns
                    summation +=\
                        self.compute_factor(label, prev_label, word) * forward_variables_matrix[t-1][label]
                forward_variables_matrix[t][label] = round(summation, 2)

        return forward_variables_matrix

    '''
    Compute the backward variables for a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: data structure containing the matrix of backward variables
    '''
    def backward_variables(self, sentence):
        sentence_length = len(sentence)
        # list of dictionaries: where the i_th dictionary represent the i_th word of the sentence and
        # each dict has entries of the form { label : backward_variable }
        backward_variables_matrix = []
        # init dictionaries
        # init dictionaries
        for i in range(0, sentence_length):
            backward_variables_matrix.append({})
            for label in self.labels:
                backward_variables_matrix[i][label] = 0.
        # init first (=last) backward variable
        for label in self.labels:
            backward_variables_matrix[sentence_length-1][label] = 1.

        for t in range(sentence_length-1, 0, -1):
            word = sentence[t][0]
            # label = sentence[t][1]
            # prev_label = sentence[t-1][1]
            for prev_label in self.labels:
                summation = 0.
                for label in self.labels:
                    # filling matrix per columns
                    summation +=\
                        self.compute_factor(label, prev_label, word) * backward_variables_matrix[t][label]
                backward_variables_matrix[t-1][prev_label] = round(summation, 2)

        return backward_variables_matrix

    # Exercise 1 b) ###################################################################
    '''
    Compute the partition function Z(x).
    Parameters: sentence: list of strings representing a sentence.
    Returns: float;
    '''
    def compute_z(self, sentence):
        forward_variables_matrix = self.forward_variables(sentence)
        sentence_length = len(sentence)
        z = 0.
        # sum over all the probabilities in the last column and return
        for label in forward_variables_matrix[sentence_length-1].keys():
            z += forward_variables_matrix[sentence_length-1][label]

        return z

    # Exercise 1 c) ###################################################################
    '''
    Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
    Parameters: sentence: list of strings representing a sentence.
                y_t: element of the set 'self.labels'; label assigned to the word at position t
                y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
                t: int; position of the word the label y_t is assigned to
    Returns: float: probability;
    '''
    def marginal_probability(self, sentence, y_t, y_t_minus_one, t):

        forward_variables_matrix = self.forward_variables(sentence)
        backward_variables_matrix = self.backward_variables(sentence)
        word = sentence[t][0]
        z = self.compute_z(sentence)
        psi = self.compute_factor(y_t, y_t_minus_one, word)
        if y_t_minus_one != 'start':
            marginal_probability = \
                (forward_variables_matrix[t-1][y_t_minus_one] * psi * backward_variables_matrix[t][y_t]) / z
        else:
            marginal_probability = (psi * backward_variables_matrix[t][y_t]) / z

        return round(marginal_probability, 2)

    # Exercise 1 d) ###################################################################
    '''
    Compute the expected feature count for the feature referenced by 'feature'
    Parameters: sentence: list of strings representing a sentence.
                feature: a feature; element of the set 'self.features'
    Returns: float: expected feature count;
    '''
    # feature here is an index! a number from feature_indices.values()
    def expected_feature_count(self, sentence, feature):

        expected_feature_count = 0.
        sentence_length = len(sentence)

        word = sentence[0][0]
        for label in self.labels:
            active_features_indices = self.get_active_features(label, 'start', word)
            if feature in active_features_indices:
                expected_feature_count += self.marginal_probability(sentence, label, 'start', 0)

        for t in range(1, sentence_length):
            for label in self.labels:
                for prev_label in self.labels:
                    active_features_indices = self.get_active_features(label, prev_label, sentence[t])
                    if feature in active_features_indices:
                        expected_feature_count += \
                                self.marginal_probability(sentence, label, prev_label, t)

        return round(expected_feature_count, 2)

    # Exercise 1 e) ###################################################################
    '''
    Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
    Parameters: word: string; a word x_i some position i of a given sentence
                label: string; the actual label of the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the empirical feature count
    '''

    def empirical_feature_count(self, label, prev_label, word):

        feature_indices_length = len(self.feature_indices)
        # init array of active features
        empirical_f_count = np.zeros(feature_indices_length)

        active_features_set = self.get_active_features(label, prev_label, word)
        for i in range (feature_indices_length):
            # check if index i is present as theta parameter in the active_features_set
            if i in active_features_set:
                empirical_f_count[i] = 1

        return empirical_f_count

    '''
    Predict the empirical feature count for a set of sentences
    Parameters: sentences: list; a list of sentences; should be a sublist of the list returned by 'import_corpus'
    Returns: (numpy) array containing the empirical feature count
    '''
    def empirical_feature_count_batch(self, sentences):

        empirical_feature_count_b = np.array([])

        for sentence in sentences:
            for i in range(0, len(sentence) - 1):
                word = sentence[i][0]
                label = sentence[i][1]

                if i == 0:
                    prev_label = 'start'
                else:
                    # takes label corresponding to the previous pair in the considered sentence
                    prev_label = sentence[i-1][1]
                empirical_feature_count = self.empirical_feature_count(label, prev_label, word)
                empirical_feature_count_b += empirical_feature_count

        return empirical_feature_count_b

    '''
    Method for training the CRF.
    Parameters: num_iterations: int; number of training iterations
                learning_rate: float
    '''
    '''
    def train(self, num_iterations, learning_rate=0.01):

        length = len(self.theta)
        expected_feature_count = np.zeros(length)
        empirical_feature_count = np.zeros(length)
        # empirical_feature_count = self.empirical_feature_count()
        empirical_feature_count_batch = self.empirical_feature_count_batch(self.corpus)

        # this trains a sentence at each iteration
        for i in range(num_iterations):
            training_sentence = random.choice(self.corpus)


            if i == 0:
                prev_label = 'start'
            else:
                prev_label = training_sentence[i-1][1]

            # feature is a tuple of two elements x_t,y_t or y_t, y_t-1
            for feature in self.feature_indices.keys():
                index = self.feature_indices[feature]
                expected_feature_count[index] = self.expected_feature_count(training_sentence, feature)

            self.theta += learning_rate * (empirical_feature_count - expected_feature_count)
    '''
    # Exercise 2 ###################################################################
    '''
    Compute the most likely sequence of labels for the words in a given sentence.
    Parameters: sentence: list of strings representing a sentence.
    Returns: list of labels; each label is an element of the set 'self.labels'
    '''
    def most_likely_label_sequence(self, sentence):

        # your code here

        pass


def main():

    corpus = import_corpus('corpus_ex.txt')
    crf = LinearChainCRF()
    crf.initialize(corpus)
    thetas = crf.get_active_features("q", "start", 'a')
    # summation = crf.get_thetas_sum(thetas)
    forward_variables = crf.forward_variables(corpus[0])  # use the first sentence
    backward_variables = crf.backward_variables(corpus[0])
    z = crf.compute_z(corpus[0])
    marginal_probability = crf.marginal_probability(corpus[0], "r", "q", 1)
    # crf.train(10)
    # CONTROL PRINTS
    # print(Colors.OKGREEN + "thetas: " + Colors.ENDC, thetas)
    # print(Colors.OKGREEN + "sum: " + Colors.ENDC, summation)
    print(Colors.OKGREEN + "forward variables matrix: " + Colors.ENDC, forward_variables)
    # for d in forward_variables:
    #     pp.pprint(d)
    print(Colors.OKGREEN + "backward variables matrix: " + Colors.ENDC, backward_variables)
    # for d in backward_variables:
    #    pp.pprint(d)
    print(Colors.OKGREEN + "Z: " + Colors.ENDC, z)
    print(Colors.OKGREEN + "marginal probability: " + Colors.ENDC, marginal_probability)
    print(crf.marginal_probability(corpus[0], "q", "start", 0))
    print(crf.marginal_probability(corpus[0], "r", "start", 0))
    print(crf.marginal_probability(corpus[0], "q", "q", 1))
    print(crf.marginal_probability(corpus[0], "q", "r", 1))
    print(crf.marginal_probability(corpus[0], "r", "r", 1))
    print(crf.marginal_probability(corpus[0], "r", "q", 1))

    for feature in crf.feature_indices.keys():
        print('feature:', feature, crf.expected_feature_count(corpus[0], crf.feature_indices[feature]))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))