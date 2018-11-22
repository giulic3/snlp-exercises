from exercise_sheet2 import *

def main():

    sentences = import_corpus("./corpus_ner.txt")
    preprocessed_sentences = preprocessing(sentences)
    initial_state_probabilities = estimate_initial_state_probabilities(preprocessed_sentences)
    transition_probabilities = estimate_transition_probabilities(preprocessed_sentences)

    ##### CONTROL PRINTS #####
    #print(bcolors.OKBLUE + 'sentences : ' + bcolors.ENDC, sentences)

    #pprint(sorted(initial_state_probabilities.items(), key=operator.itemgetter(1), reverse=True))

    return

if __name__ == "__main__":
    main()
