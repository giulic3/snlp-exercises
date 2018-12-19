################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import numpy as np
import random
from random import randint
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

    # number of words used for training
    # this var is needed to implement w_b in exercise 5)
    num_training_words = None

    # Exercise 1 a) ###################################################################
    '''
    Initialize the maximum entropy model, i.e., build the set of all features, the set of all labels
    and create an initial array 'theta' for the parameters of the model.
    Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
    '''

    def initialize(self, corpus):
        # (the, DT), (dog, NN)
        self.corpus = corpus
        self.feature_indices = {}
        self.labels = set()
        self.num_training_words = 0
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
        self.theta = np.ones((len(F_list),), dtype=float)

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

    def cond_normalization_factor(self, word, prev_label):
        z = 0
        for label in self.labels:
            x = 0
            active_features = self.get_active_features(word, label, prev_label)
            x += np.dot(self.theta, active_features)
            z += math.exp(x)

        # print(Colors.OKBLUE + "1/Z(" + word + "): " + Colors.ENDC, np.reciprocal(z))
        return np.reciprocal(z)

    # Exercise 2 b) ###################################################################
    '''
     Compute the conditional probability of a label given a word x_i.
     Parameters: label: string; we are interested in the conditional probability of this label
                 word: string; a word x_i at some position i of a given sentence
                 prev_label: string; the label of the word at position i-1
     Returns: float
     '''

    def conditional_probability(self, label, word, prev_label):

        active_features = self.get_active_features(word, label, prev_label)
        x = np.dot(self.theta, active_features)
        conditional_probability = self.cond_normalization_factor(word, prev_label) * math.exp(x)

        # print(Colors.OKBLUE + "conditional_probability: " + Colors.ENDC, conditional_probability)
        return conditional_probability

    # Exercise 3 a) ###################################################################
    '''
    Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
    Parameters: word: string; a word x_i some position i of a given sentence
                label: string; the actual label of the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the empirical feature count
    '''

    def empirical_feature_count(self, word, label, prev_label):

        return self.get_active_features(word, label, prev_label)

    # Exercise 3 b) ###################################################################
    '''
    Compute the expected feature count given a word, the label of the previous word and the parameters of the current
    model (see variable theta)
    Parameters: word: string; a word x_i at some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the expected feature count
    '''
    def expected_feature_count(self, word, prev_label):

        expected_f_count = np.zeros(len(self.feature_indices))

        for feature in self.feature_indices:
            for label in self.labels:
                conditional_probability = self.conditional_probability(label, word, prev_label)
                active_features = self.get_active_features(word, label, prev_label)
                active_feature_value = active_features[self.feature_indices[(word, label)]]

                expected_f_i_count = conditional_probability * active_feature_value
                expected_f_count[self.feature_indices[feature]] = expected_f_i_count

        # print(Colors.OKBLUE + "expected_f_count: " + Colors.ENDC, expected_f_count)
        return expected_f_count

    # Exercise 4 a) ###################################################################
    '''
    Do one learning step.
    Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                label: string; the actual label of the selected word
                prev_label: string; the label of the word at position i-1
                learning_rate: float
    '''

    def parameter_update(self, word, label, prev_label, learning_rate):
        # Numpy arrays diffs element by element
        feature_count_difference = \
            self.empirical_feature_count(word, label, prev_label) - self.expected_feature_count(word, prev_label)
        alpha_factor = learning_rate * feature_count_difference
        # Update theta parameters
        self.theta += alpha_factor

    # Exercise 4 b) ###################################################################
    '''
    Implement the training procedure.
    Parameters: number_iterations: int; number of parameter updates to do
                learning_rate: float
    '''

    def train(self, number_iterations, learning_rate=0.1):

        for j in range(number_iterations):
            training_sentence = random.choice(self.corpus)
            # Needed for w_b in exercise 5)
            self.num_training_words = len(training_sentence)
            i = randint(0, len(training_sentence) - 1)
            training_pair = training_sentence[i]
            word = training_pair[0]
            label = training_pair[1]

            if i == 0:
                prev_label = 'start'
            else:
                prev_label = training_sentence[i-1][1]

            self.parameter_update(word, label, prev_label, learning_rate)
            # print(Colors.OKBLUE + "theta: " + Colors.ENDC, self.theta)

    # Exercise 4 c) ###################################################################
    '''
     Predict the most probable label of the word referenced by 'word'
     Parameters: word: string; a word x_i at some position i of a given sentence
                 prev_label: string; the label of the word at position i-1
     Returns: string; most probable label
     '''
    def predict(self, word, prev_label):

        # Compute all the conditional probabilities given x_i e take the maximum
        conditional_probabilities = np.array([])
        args_labels = np.array([])
        for label in self.labels:
            prob = self.conditional_probability(label, word, prev_label)
            conditional_probabilities = np.append(conditional_probabilities, prob)
            args_labels = np.append(args_labels, label)

        label_index = np.argmax(conditional_probabilities)
        most_probable_label = args_labels[label_index]

        return most_probable_label

    # Exercise 5 a) ###################################################################
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
                    # Takes label corresponding to the previous pair in the considered sentence
                    prev_label = sentence[i-1][1]
                empirical_feature_count = self.empirical_feature_count(word, label, prev_label)
                empirical_feature_count_b += empirical_feature_count

        return empirical_feature_count_b

    # Exercise 5 a) ###################################################################
    '''
    Predict the expected feature count for a set of sentences
    Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
    Returns: (numpy) array containing the expected feature count
    '''

    def expected_feature_count_batch(self, sentences):
        expected_feature_count_b = np.array([])

        for sentence in sentences:
            for i in range(0, len(sentence) - 1):
                word = sentence[i][0]
                # label = sentence[i][1]
                if i == 0:
                    prev_label = 'start'
                else:
                    # Takes label corresponding to the previous pair in the considered sentence
                    prev_label = sentence[i-1][1]
                expected_feature_count = self.expected_feature_count(word, prev_label)
                expected_feature_count_b += expected_feature_count

        return expected_feature_count_b

    # Exercise 5 b) ###################################################################
    '''
    Implement the training procedure which uses 'batch_size' sentences from to training corpus
    to compute the gradient.
    Parameters: number_iterations: int; number of parameter updates to do
                batch_size: int; number of sentences to use in each iteration
                learning_rate: float
    '''
    def train_batch(self, number_iterations, batch_size, learning_rate=0.1):

        if batch_size >= len(self.corpus):
            print('Error! batch_size is bigger than the corpus size!')
        else:
            for it in range(number_iterations):
                self.train(batch_size, learning_rate)

    '''
    Getter method for num_training_words member.
    '''
    def get_num_training_words(self):

        return self.num_training_words
# Exercise 5 c) ###################################################################


'''
Compare the training methods 'train' and 'train_batch' in terms of convergence rate
Parameters: corpus: list of list; a corpus returned by 'import_corpus'
'''


def evaluate(corpus):

    N = 1  # must be large to ensure convergence
    w_a_tmp = 0
    w_b_tmp = 0
    # counters that save the number of label predictions given by predict() function
    correct_predictions_a = 0
    correct_predictions_b = 0
    # numpy arrays that store accuracy scores over a number of iterations
    accuracy_a = np.array([])
    accuracy_b = np.array([])
    # numpy arrays that store number of words used for training over a num of iterations
    w_a = np.array([])
    w_b = np.array([])
    # create test set by selecting 10% of all sentences randomly
    np.random.shuffle(corpus)
    corpus_length = len(corpus)
    test_set_size = int(round(corpus_length / 10, 0))
    training_set_size = int(corpus_length - test_set_size)
    training_set, test_set = corpus[:training_set_size], corpus[test_set_size:]
    num_predictions_so_far = 0
    # create instance A of MaxEntModel to be used with train()
    A = MaxEntModel()
    A.initialize(training_set)
    # create instance B of MaxEntModel to be used with train_batch()
    B = MaxEntModel()
    B.initialize(training_set)
    # train A and B
    for i in range(N):
        print(Colors.WARNING + "Iteration: " + Colors.ENDC, i)
        print(Colors.WARNING + "Training A..." + Colors.ENDC)
        A.train(1)
        w_a_tmp += 1
        print(Colors.WARNING + "Training B..." + Colors.ENDC)
        B.train_batch(1, 1)
        w_b_tmp += B.get_num_training_words()

        # execute predict on the test_set
        for sentence in test_set:
            for j in range(len(test_set)):
                word = sentence[j][0]
                label = sentence[j][1]
                if j == 0:
                    prev_label = 'start'
                else:
                    prev_label = sentence[j-1][1]
                print(Colors.WARNING + "Predicting label for A..." + Colors.ENDC)
                prediction_a = A.predict(word, prev_label)
                print(Colors.WARNING + "Predicting label for B..." + Colors.ENDC)
                prediction_b = B.predict(word, prev_label)
                num_predictions_so_far += 1

                if prediction_a == label:
                    correct_predictions_a += 1

                if prediction_b == label:
                    correct_predictions_b += 1
                # compute accuracy for model A and B
                it_accuracy_a = correct_predictions_a / num_predictions_so_far
                it_accuracy_b = correct_predictions_b / num_predictions_so_far

                accuracy_a = np.append(accuracy_a, it_accuracy_a)
                accuracy_b = np.append(accuracy_b, it_accuracy_b)
                w_a = np.append(w_a, w_a_tmp)
                w_b = np.append(w_b, w_b_tmp)
    # plot the data (accuracy against number of words)
    print(Colors.WARNING + "Plotting data..." + Colors.ENDC)
    chart_a = plt.plot(w_a, accuracy_a)
    chart_b = plt.plot(w_b, accuracy_b)
    plt.setp(chart_a, color='r', linewidth=2.0)
    plt.setp(chart_b, color='b', linewidth=2.0)
    # save the plot on file
    pp = PdfPages('plot.pdf')
    pp.savefig()
    pp.close()
