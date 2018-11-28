################################################################################
## SNLP exercise sheet 2
################################################################################

from collections import Counter
import time, operator
from operator import itemgetter
from pprint import pprint
from math import log

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
Returns: list of list; the second layer list contains tuples (token,label);
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


'''
Replace tokens occurring only once in the corpus by the token <unknown>.
Parameters:	sentences; list of lists (where each element is a tuple)
Returns: list of lists; a new data structure containing the sentences transformed after the preprocessing
'''


def preprocessing(sentences):
    # First count number of occurrences for each token
    # { token : number_of_occurrences }
    occurrences = Counter()
    preprocessed_sentences = []
    mod_sentence = []

    for sentence in sentences:
        for tuple in sentence:
            occurrences[tuple[0]] += 1

    # Then replace with token <unknown> when occurrence = 1
    for sentence in sentences:
        for tuple in sentence:
            num_of_occurrences = occurrences[tuple[0]]
            if num_of_occurrences == 1:
                mod_sentence.append(("<unknown>", tuple[1]))
            else:
                mod_sentence.append((tuple[0], tuple[1]))
        preprocessed_sentences.append(mod_sentence)
        mod_sentence = []

    return preprocessed_sentences


# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parameterization of this probability distribution;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''


def initial_state_probabilities(state, internal_representation):
    return internal_representation[state]


'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''


def transition_probabilities(from_state, to_state, internal_representation):
    return internal_representation[from_state][to_state]


'''
Implement the matrix of emission probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''


def emission_probabilities(state, emission_symbol, internal_representation):
    return internal_representation[state][emission_symbol]


'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''


def estimate_initial_state_probabilities(corpus):
    # Count the frequencies for the states/tokens at the beginning of a sentence
    frequencies = Counter()
    # { state/token : initial probability }
    initial_state_probabilities = {}
    sum_frequencies = len(corpus)

    for sentence in corpus:
        first_tuple = sentence[0]
        frequencies[first_tuple[1]] += 1

    for sentence in corpus:
        first_tuple = sentence[0]
        # If a key is missing that means that the associated probability is zero!
        initial_state_probabilities[first_tuple[1]] = frequencies[first_tuple[1]] / float(sum_frequencies)

    return initial_state_probabilities


# TODO do I need the probability to go from state to eos?
'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''


def estimate_transition_probabilities(corpus):
    # Frequencies dict { s1 : { sj : freq }}
    transition_probabilities = {}
    state_frequencies = Counter()
    # Init outer level of the dict
    for sentence in corpus:
        for tuple in sentence:
            transition_probabilities[tuple[1]] = {}
    # pprint(transition_probabilities)
    # Compute frequencies
    for sentence in corpus:
        for i in range (len(sentence)-1):
            if sentence[i+1][1] in transition_probabilities[sentence[i][1]]:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]] += 1
            else:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]] = 1

    # TODO remove from the count the ending states! namely, use the sum of single freqs to obtain the total
    # Count total frequencies for each label
    for sentence in corpus:
        for tuple in sentence:
            state_frequencies[tuple[1]] += 1
    # print(Colors.OKBLUE + 'state_frequencies: ' + Colors.ENDC, state_frequencies)

    for s_i in transition_probabilities:
        # print(Colors.OKGREEN+'k/v ='+Colors.ENDC, s_i, ' : ', transition_probabilities[s_i])
        for s_j in transition_probabilities[s_i]:
            # print('s_j', s_j)
            # If a key is missing that means that the associated probability is zero!
            # print(True)
            transition_probabilities[s_i][s_j] /= float(state_frequencies[s_i])

    return transition_probabilities


'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''


def estimate_emission_probabilities(corpus):
    # Dict { label/state : { observation:prob }
    emission_probabilities = {}
    state_frequencies = Counter()

    # Init outer level of the dict
    for sentence in corpus:
        for tuple in sentence:
            emission_probabilities[tuple[1]] = {}

    # Compute frequencies
    for sentence in corpus:
        for i in range (len(sentence)-1):
            if sentence[i+1][0] in emission_probabilities[sentence[i][1]]:
                emission_probabilities[sentence[i][1]][sentence[i+1][0]] += 1
            else:
                emission_probabilities[sentence[i][1]][sentence[i+1][0]] = 1

    # Count total frequencies for each label
    for sentence in corpus:
        for tuple in sentence:
            state_frequencies[tuple[1]] += 1

    for s in emission_probabilities:
        # print(Colors.OKGREEN+'k/v ='+Colors.ENDC, s, ' : ', emission_probabilities[s])
        for obs in emission_probabilities[s]:
            # If a key is missing that means that the associated probability is zero!
            emission_probabilities[s][obs] /= float(state_frequencies[s])

    # print('emission probabilities: ')
    # pprint(emission_probabilities)

    return emission_probabilities


'''
Parameters: corpus: list of lists (where each element is a tuple); the corpus of sentences
Returns: list; list containing of all possible states in the corpus
'''


def get_states_set(corpus):
    S = set()
    states = []

    for sentence in corpus:
        for tuple in sentence:
            S.add(tuple[1])

    for item in S:
        states.append(item)

    return states


'''
Parameters: corpus: list of strings; the corpus of sentences AFTER preprocessing
Returns: list; list containing of all possible observations in the corpus
'''


def get_observations_set(corpus):
    O = set()
    observations = []

    for sentence in corpus:
        for tuple in sentence:
            O.add(tuple[0])

    for item in O:
        observations.append(item)

    return observations


# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_symbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing
            the parameters of the probability distribution of the initial states,
            returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters
            of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters
            of the matrix of emission probabilities, returned by estimate_emission_probabilities
            states: set of all possible states according to the model
            observations: set of all possible observations according to the model
Returns: list of strings; the most likely state sequence
'''


def most_likely_state_sequence(observed_symbols, initial_state_probabilities_parameters,
                               transition_probabilities_parameters, emission_probabilities_parameters, states, observations):
    # List of maximum values (probabilities)
    delta = []
    # List of argmax values (states corresponding to probabilities)
    phi = []
    # Init trellis matrix
    num_rows, num_columns = len(states), len(observed_symbols)
    trellis = [[float(0) for j in range(num_columns)] for i in range(num_rows)]

    # print("observed_symbols BEFORE pre-processing: ", observed_symbols)
    for index, word in enumerate(observed_symbols):
        if word not in observations:
            observed_symbols[index] = "<unknown>"

    print("observed_symbols AFTER pre-processing: ", observed_symbols)

    print(Colors.OKGREEN + 'trellis BEFORE: ' + Colors.ENDC)
    pprint(trellis)

    for i in range(num_rows):
        try:
            trellis[i][0] = log(initial_state_probabilities_parameters[states[i]])
        except KeyError as e:
            print(Colors.WARNING + 'KeyError exception, initial_state_probabilities_parameters' + Colors.ENDC, e)

    for k in range(1, num_columns):  # symbols

        tmp_max_column = []
        tmp_argmax_column = []

        for i in range(num_rows):  # states s_i
            for j in range(num_rows):  # states s_j
                try:
                    a = log(transition_probabilities_parameters[states[j]][states[i]])
                except KeyError as e:
                    # print(Colors.WARNING + 'KeyError exception, in computing trellis' + Colors.ENDC, e)
                    a = 0
                    # a = -30000
                try:
                    b = log(emission_probabilities_parameters[states[i]][observed_symbols[k]])
                except KeyError as e:
                    # print(Colors.WARNING + 'KeyError exception, in computing trellis' + Colors.ENDC, e)
                    b = 0
                    # b = -30000
                tmp_max_column.append(trellis[j][k-1] + a + b)

            trellis[i][k] = max(tmp_max_column)
            index = tmp_max_column.index(trellis[i][k])
            # TODO this line doesn't work!
            #index, trellis[i][k] = max(enumerate(tmp_max_column), key=itemgetter(1))
            tmp_argmax_column.append(states[index])  # this gives me the argmax, namely the label/state

        phi.append(max([item for item in tmp_argmax_column]))

    print(Colors.OKGREEN + 'trellis AFTER: ' + Colors.ENDC)
    pprint(trellis)
    print(Colors.OKGREEN + 'delta: ' + Colors.ENDC, delta)
    print(Colors.OKGREEN + 'phi: ' + Colors.ENDC, phi)

    return phi
