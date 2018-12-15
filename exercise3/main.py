from exercise_sheet3 import *


def main():

    max_entropy_object = MaxEntModel()
    max_entropy_object.get_active_features("the", "DT", "start")
    # max_entropy_object.cond_normalization_factor("the", "start")
    # max_entropy_object.conditional_probability("the", "DT", "start")
    max_entropy_object.expected_feature_count("the", "start")


if __name__ == "__main__":
    main()

