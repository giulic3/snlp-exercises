from exercise_sheet2 import *

def main():

    sentences = import_corpus("./corpus_ner.txt")
    preprocessed_sentences = preprocessing(sentences)
    initial_state_probabilities = estimate_initial_state_probabilities(preprocessed_sentences)
    #initial_state_probabilities()
    #transition_probabilities()
    #emission_probabilities()

    #print(sentences)
    '''
    for i in range(10):
        print("s:",sentences[i])
        print("ps:",preprocessed_sentences[i])
        time.sleep(3)
    '''
    #pprint(sorted(initial_state_probabilities.items(), key=operator.itemgetter(1), reverse=True))

    return

if __name__ == "__main__":
    main()
