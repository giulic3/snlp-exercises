from exercise_sheet3 import *


def main():

    corpus = import_corpus('./corpus_fake.txt')
    max_entropy_object = MaxEntModel()
    max_entropy_object.initialize(corpus)
    # max_entropy_object.get_active_features("the", "DT", "start")
    # max_entropy_object.cond_normalization_factor("the", "start")
    # max_entropy_object.conditional_probability("the", "DT", "start")
    # max_entropy_object.expected_feature_count("the", "start")
    # max_entropy_object.parameter_update("the", "DT", "start", 0.1)
    # max_entropy_object.train(2)
    # label_prediction = max_entropy_object.predict('the', 'start')
    # max_entropy_object.empirical_feature_count_batch(corpus[0:2])
    evaluate(corpus)
    # print(Colors.OKGREEN + "label_prediction: " + Colors.ENDC, label_prediction)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))


