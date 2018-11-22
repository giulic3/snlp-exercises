from exercise_sheet2 import *

def main():

    sentences = import_corpus("./corpus_ner.txt")
    preprocessed_sentences = preprocessing(sentences)
    initial_state_probabilities = estimate_initial_state_probabilities(preprocessed_sentences)
    transition_probabilities = estimate_transition_probabilities(preprocessed_sentences)
    emission_probabilities = estimate_emission_probabilities(preprocessed_sentences)
    ##### CONTROL PRINTS #####
    #print(bcolors.OKBLUE + 'sentences : ' + bcolors.ENDC, sentences)

    #pprint(sorted(initial_state_probabilities.items(), key=operator.itemgetter(1), reverse=True))
    #print(bcolors.OKBLUE + 'transition_probabilities: ' + bcolors.ENDC)
    #pprint(transition_probabilities)

    return

if __name__ == "__main__":
    main()
