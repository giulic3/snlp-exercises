from exercise_sheet2 import *


def main():

    sentences = import_corpus("./corpus_ner.txt")

    # Training
    preprocessed_sentences = preprocessing(sentences)
    initial_state_probabilities = estimate_initial_state_probabilities(preprocessed_sentences)
    transition_probabilities = estimate_transition_probabilities(preprocessed_sentences)
    emission_probabilities = estimate_emission_probabilities(preprocessed_sentences)
    states = get_states_set(preprocessed_sentences)
    observations = get_observations_set(preprocessed_sentences)

    # Testing
    test_sentence_list = get_test_sentence(preprocessed_sentences, 14)
    observed_sentence = [tuple[0] for tuple in test_sentence_list]

    viterbi_state_sequence = most_likely_state_sequence(
        observed_sentence, initial_state_probabilities, transition_probabilities,
        emission_probabilities, states, observations)

    # Control prints
    print(Colors.OKBLUE + 'observed_sentence after pre-processing : ' + Colors.ENDC, observed_sentence)
    print(Colors.OKBLUE + 'viterbi_state_sequence: ' + Colors.ENDC, viterbi_state_sequence)

    return


if __name__ == "__main__":
    main()
