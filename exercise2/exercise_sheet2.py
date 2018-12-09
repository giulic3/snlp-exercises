################################################################################
## SNLP exercise sheet 2
################################################################################

from collections import Counter
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
                # Tranform all observations into lowercase
                mod_sentence.append((tuple[0].lower(), tuple[1]))
                # mod_sentence.append((tuple[0], tuple[1]))

        preprocessed_sentences.append(mod_sentence)
        mod_sentence = []

    return preprocessed_sentences


'''
Parameters: sentences: list of lists; the corpus of sentences AFTER pre-processing
            n: number of pops on the sentences list
Returns: list; list containing the sentence on which to perform the test
'''


def get_test_sentence(sentences, n):
    last_sentence = []

    while n > 0:
        last_sentence = sentences.pop()
        n = n - 1

    return last_sentence


# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parametrization of this probability distribution;
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

    # Compute frequencies
    for sentence in corpus:
        for i in range(len(sentence)-1):
            if sentence[i+1][1] in transition_probabilities[sentence[i][1]]:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]] += 1
            else:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]] = 1

    # Count total frequencies for each label
    for sentence in corpus:
        for tuple in sentence:
            state_frequencies[tuple[1]] += 1

    for s_i in transition_probabilities:
        for s_j in transition_probabilities[s_i]:
            # If a key is missing that means that the associated probability is zero!
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
        for i in range(len(sentence)):
            if sentence[i][0] in emission_probabilities[sentence[i][1]]:
                emission_probabilities[sentence[i][1]][sentence[i][0]] += 1
            else:
                emission_probabilities[sentence[i][1]][sentence[i][0]] = 1

    # Count total frequencies for each label
    for sentence in corpus:
        for tuple in sentence:
            state_frequencies[tuple[1]] += 1

    for s in emission_probabilities:
        for obs in emission_probabilities[s]:
            # If a key is missing that means that the associated probability is zero!
            emission_probabilities[s][obs] /= float(state_frequencies[s])

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
    # List of argmax values (states corresponding to probabilities)
    phi = []
    # Init trellis matrix
    num_rows, num_columns = len(states), len(observed_symbols)
    trellis = [[float(0) for j in range(num_columns)] for i in range(num_rows)]

    for index, word in enumerate(observed_symbols):
        if word not in observations:
            observed_symbols[index] = "<unknown>"

    for i in range(num_rows):
        try:
            trellis[i][0] = log(initial_state_probabilities_parameters[states[i]])
        except KeyError as e:
            # print(Colors.WARNING + 'KeyError exception, initial_state_probabilities_parameters' + Colors.ENDC, e)
            trellis[i][0] = float("-inf")

    trellis_first_column = [trellis[i][0] for i in range(num_rows)]
    phi.append(states[trellis_first_column.index(max(trellis_first_column))])

    for k in range(1, num_columns):  # symbols

        delta = []

        for i in range(num_rows):  # states s_i
            tmp_max_column = []
            for j in range(num_rows):  # states s_j

                try:
                    a = log(transition_probabilities_parameters[states[j]][states[i]])
                except KeyError as e:
                    a = float("-inf")

                try:
                    b = log(emission_probabilities_parameters[states[i]][observed_symbols[k]])
                except KeyError as e:
                    b = float("-inf")

                tmp_max_column.append(trellis[j][k-1] + a + b)
            # Store in the trellis the max value for each s_j, given a s_i
            trellis[i][k] = max(tmp_max_column)
            delta.append(trellis[i][k])

        # Store in phi the label/state corresponding to the index associated to the max value
        phi.append(states[delta.index(max(delta))])

    return phi

