import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def RandomizedGridSearch(n_experiments,
                           pipe,
                           param_distributions,
                           train_X,
                           train_y,
                           test_X,
                           test_y,
                           scoring='neg_mean_squared_error'):
    '''
    My own version of RandomizedSearchCV
    :param n_experiments: Number of randomized experiments to run
    :param pipe: scikit-learn pipeline
    :param param_distributions: scikit-learn parameter grid
    :param train_X: numpy
    :param train_y: numpy
    :param test_X: numpy
    :param test_y: numpy
    :param scoring: string method
    :return: experiment results dataframe
    '''
    # Copy data
    train_X, train_y = train_X.copy(), train_y.copy()
    test_X, test_y = test_X.copy(), test_y.copy()

    # Get transform_class_list
    transform_class_list = []
    for class_key, feature_is_included_list in param_distributions.items():
        transform_class_list.append(pipe.named_steps[class_key])

    # Initialize experiments dictionary
    experiments_info = {}
    for class_key, feature_is_included_list, transform_class in \
        zip(param_distributions.items(), transform_class_list):
        for i in range(len(feature_is_included_list)):
            experiments_info[class_key + "___" + str(i)] = []
    experiments_info['score'] = []

    # Iterate over the experiments
    for iteration in tqdm(range(n_experiments)):

        # Updates the transform parameters
        for class_key, feature_is_included_list, transform_class in \
            zip(param_distributions.items(), transform_class_list):

            # Copy feature_is_included_list
            feature_is_included_list = feature_is_included_list.copy()

            # Loop over feature_is_included_list
            for feature_i in range(len(feature_is_included_list)):

                # Replace feature_is_included_list
                feature_is_included_list[feature_i] = np.random.choice([
                    True, False
                ]) if feature_is_included_list[feature_i] == None else feature_is_included_list[feature_i]

                # Save input data for the experiments dataframe output
                experiments_info[class_key + "___" +
                                 str(feature_i)].append(feature_is_included_list[feature_i])

            # Set parameters for the transformation class (typically numeric fields)
            setattr(transform_class, "feature_is_included_list", feature_is_included_list)

        # Fit
        pipe.fit(train_X, train_y)

        # Predict
        pred_y = pipe.predict(test_X)

        # Scoring
        if scoring == 'neg_mean_squared_error':
            score = mean_squared_error(pred_y, test_y)
        else:
            raise Exception('Scoring type not implemented')

        # Appending the score
        experiments_info["score"].append(score)

    experiments_results = pd.DataFrame(experiments_info)
    return experiments_results
