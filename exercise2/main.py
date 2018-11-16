from exercise_sheet2 import *

def main():
    # a list of list
    sentences = []

    sentences = import_corpus("./corpus_ner.txt")
    preprocessing(sentences)
    #initial_state_probabilities()
    #transition_probabilities()
    #emission_probabilities()

    #print(sentences)

    return

if __name__ == "__main__":
    main()
