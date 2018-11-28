from exercise_sheet2 import *


def main():

    sentences = import_corpus("./corpus_ner.txt")
    # sentences = import_corpus("./corpus_gattocane.txt")
    last_sentence = sentences.pop()
    observed_sentence = [tuple[0] for tuple in last_sentence]
    print(Colors.OKBLUE + 'observed_sentence : ' + Colors.ENDC, observed_sentence)

    preprocessed_sentences = preprocessing(sentences)
    initial_state_probabilities = estimate_initial_state_probabilities(preprocessed_sentences)
    transition_probabilities = estimate_transition_probabilities(preprocessed_sentences)
    emission_probabilities = estimate_emission_probabilities(preprocessed_sentences)
    states = get_states_set(preprocessed_sentences)
    observations = get_observations_set(preprocessed_sentences)


    viterbi_state_sequence = most_likely_state_sequence(
        observed_sentence, initial_state_probabilities, transition_probabilities,
        emission_probabilities, states, observations)

    # CONTROL PRINTS
    print(Colors.OKBLUE + 'observed_sentence : ' + Colors.ENDC, observed_sentence)
    print(Colors.OKBLUE + 'initial_state_probabilities: ' + Colors.ENDC)
    # pprint(sorted(initial_state_probabilities.items(), key=operator.itemgetter(1), reverse=True))
    print(Colors.OKBLUE + 'transition_probabilities: ' + Colors.ENDC)
    # pprint(transition_probabilities)
    print(Colors.OKBLUE + 'emission_probabilities: ' + Colors.ENDC)
    # pprint(emission_probabilities)
    print('viterbi_state_sequence: ', viterbi_state_sequence)

    return


if __name__ == "__main__":
    main()
