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


    # num_features
    num_features = len(param_distributions['standard_scaler'])

    # initialize experiments_info
    experiments_info = {}
    for transform in param_distributions.keys():
        for i in range(num_features):
            experiments_info[transform + "__" + str(i)] = []
    experiments_info['score'] = []

    # Iterate over the experiments
    for iteration in tqdm(range(n_experiments)):

        for transform, priors in param_distributions.items():

            for index in range(num_features):

                probability_of_true = priors[index]
                if np.random.uniform() < probability_of_true:
                    True

                # Random choice
                if transform_selection == None:
                    transform_selection = np.random.choice(list(options[transform_category].keys())) + "__true"

                # Split
                selection_type = transform_selection.split("__")[0]
                selection_value = True if transform_selection.split("__")[1].lower() == 'true' else False

                # Set all selections to False if a True is found
                if selection_value == True:
                    for transform_selection_possibility in options[transform_category].keys():
                        options[transform_category][transform_selection_possibility][index] = False

                # Now, set the chosen selection to the right value
                options[transform_category][selection_type][index] = selection_value


        # i.e. scalers
        for transform_category in options.keys():

            # i.e. min_max_scaler
            for transform_selection_possibility in options[transform_category]:

                # i.e. MinMaxScaler()
                transform_class = pipe.named_steps[transform_selection_possibility]

                # Get feature_is_included_list from options
                feature_is_included_list = options[transform_category][transform_selection_possibility]

                # Copy so any changes aren't saved
                feature_is_included_list = feature_is_included_list.copy()

                # Loop over feature indices
                for feature_i in range(num_features):

                    # Save input data for the experiments dataframe output
                    experiments_info[transform_category + "__" + transform_selection_possibility + "__" + str(feature_i)].append(feature_is_included_list[feature_i])

                # Set the attribute
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

    # Return pandas experiments_results
    return pd.DataFrame(experiments_info).sort_values(by=['score'], ascending=False, inplace=False)
