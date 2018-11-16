################################################################################
## SNLP exercise sheet 2
################################################################################

from collections import Counter
import time, operator
from pprint import pprint


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
    internal_representation = {}

    for sentence in corpus:
        first_tuple = sentence[0]
        frequencies[first_tuple[0]] += 1
    sum_frequencies = len(corpus)

    for sentence in corpus:
        first_tuple = sentence[0]
        # If a key is missing that means that the associated probability is zero!
        internal_representation[first_tuple[0]] = frequencies[first_tuple[0]] / float(sum_frequencies)

    return internal_representation




'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus):
    # Dizionario di frequenze { s1 : { sj : freq }}




'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus):
    pass






# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_symbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_symbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):
    pass
