

import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV

def sample_uniq_vals_from_series(
    num_uniq_values: int,
    column: pd.Series,
    random_state: int = None
) -> list:
    """
    Sample unique values from a pandas Series.

    Parameters
    ----------
    num_uniq_values : int
        Maximum number of unique values to sample.
    column : pd.Series
        Series from which values are sampled.
    random_state : int or None, default=None
        Seed for reproducible sampling.

    Returns
    -------
    list
        A list containing sampled unique values. At most one missing
        value is included.

    Examples
    --------
    >>> sample_uniq_vals_from_series(
    ...     5,
    ...     xmas_movies["genre"],
    ...     random_state=123
    ... )
    """
    if not isinstance(column, pd.Series):
        raise TypeError("column must be a pandas Series.")

    if not isinstance(num_uniq_values, int):
        raise TypeError("num_uniq_values must be an integer.")

    if num_uniq_values < 0:
        raise ValueError("num_uniq_values cannot be negative.")

    unique_values = column.drop_duplicates()

    sample_size = min(num_uniq_values, len(unique_values))

    sampled_values = unique_values.sample(
        n=sample_size,
        random_state=random_state
    )

    return sampled_values.tolist()


def get_sample_vals_from_each_col( num_uniq_values: int, frame: pd.DataFrame, random_state: int = None) :

    """
    Sample unique values from each col of a pandas DataFrame

    Parameters
    ----------
    num_uniq_values : int
        Maximum number of unique values to sample.
    frame : pd.DataFrame
        DataFrame from which values are sampled.
    random_state : int or None, default=None
        Seed for reproducible sampling.

    Returns
    -------
    None. The function displays sample unique values from each column.

    Notes
    ------
    The function applies sample_uniq_vals_from_series to each column
    in the provided DataFrame, spcified via `frame` parameter.
    """

    for c in frame.columns :
        print(f"Unique values from {c} :\n", '\n', sample_uniq_vals_from_series(num_uniq_values, frame[c], random_state), '\n' )

def get_nans_overview(frame, missing_only=True, decimals=3,ascending=False):
    """
    Summarize missing values in each column of a DataFrame.

    Parameters
    ----------
    frame : pandas.DataFrame
        DataFrame to inspect.
    missing_only : bool, default=True
        If True, include only columns with at least one missing value.
    decimals : int, default=3
        Number of decimal places used to round the results.
    ascending : bool, default=False
        If False, the columns with the most missing values will be displayed on top.

    Returns
    -------
    pandas.DataFrame
        Number and fraction of missing values in each included column,
        sorted by the number of missing values, by default in descending order.
    """
    nans_absolute = frame.isna().sum().sort_values(ascending=ascending)
    nans_relative = nans_absolute.div(frame.shape[0])

    nan_frame = pd.concat(
        [nans_absolute, nans_relative], axis=1
    ).rename(columns={
        0: "Number of NaNs",
        1: "Fraction of NaNs",
    })

    if missing_only:
        nan_frame = nan_frame[nan_frame["Fraction of NaNs"] > 0]

    return nan_frame.round(decimals)

def convert_probab_preds_to_binary_preds(predicted_probabs, treshold = 0.5, name = 'Downtime') :
    """
    Convert predicted probabilities to binary predictions using a given threshold.

    Example
    --------
    y_pred[ : 5]

        array([0.00373895, 0.99699745, 0.99717775, 0.00335553, 0.00335707])

    convert_probab_preds_to_binary_preds(y_pred).head()

        0    0
        1    1
        2    1
        3    0
        4    0
        Name: Downtime, dtype: int64
    """
    return pd.Series( [ 1 if p > treshold else 0 for p in predicted_probabs ], name = name )




def make_classification_report_given_treshold(y_test, predicted_probabs, treshold = 0.5, name = 'Downtime') :

    binary_probabs = convert_probab_preds_to_binary_preds(predicted_probabs, treshold, name)
    report = classification_report(y_test, binary_probabs)

    treshold_probab = treshold * 100

    print(f'Model performance when treshold probability is {treshold_probab:0.1f} %', '\n' )
    print(report)



def get_model_predictions_plus_stats(X_train, y_train, X_test, y_test, final_model, ret_probs = False) :
    """
    Generate probabilities and print evaluation metrics for a binary classifier.

    Print ROC AUC and accuracy for the training and test sets, plus a test
    classification report at a threshold of 0.5 and test average precision.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Feature matrix for the training set.
    y_train : pandas.Series
        True binary training labels, encoded as 0 and 1.
    X_test : pandas.DataFrame
        Feature matrix for the test set.
    y_test : pandas.Series
        True binary test labels, encoded as 0 and 1.
    final_model : estimator object
        A fitted binary classifier. If predict_proba is available, its second
        column is used and assumed to correspond to class 1. Otherwise, predict
        must return positive-class probabilities, as with a native LightGBM
        Booster using a binary classification objective.
    ret_probs : bool, default=False
        If True, return positive-class probabilities. Otherwise, return
        binary predictions using a threshold of 0.5.

    Returns
    -------
    predictions : numpy.ndarray or pandas.Series
        Positive-class probabilities when ret_probs=True; otherwise,
        binary predictions for the test set.

    Notes
    -----
    ROC AUC and average precision are calculated from probabilities.
    Accuracy and the classification report use binary predictions obtained
    with a threshold of 0.5.

    Requires the helper functions:
    - convert_probab_preds_to_binary_preds
    - make_classification_report_given_treshold

    Examples
    --------
    >>> y_probs = get_model_predictions_plus_stats(
    ...     X_train, y_train, X_test, y_test, model, ret_probs=True
    ... )
    Train AUC is 0.947 and Test AUC is 0.912
    Train accuracy is 0.873 and Test accuracy is 0.846
    <classification report printed here>
    Average test precision score is 0.735
    """

    def run_predictions(X, model):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    
    y_pred = run_predictions(X_test, final_model)
    y_pred_binary = convert_probab_preds_to_binary_preds(y_pred)
    in_sample_pred = run_predictions(X_train, final_model)
    in_sample_pred_binary = convert_probab_preds_to_binary_preds(in_sample_pred)
    train_auc = roc_auc_score(y_train, in_sample_pred)
    test_auc = roc_auc_score(y_test, y_pred)
    print(f"Train AUC is {train_auc:0.3f} and Test AUC is {test_auc:0.3f}", '\n')
    train_acc = accuracy_score(y_train, in_sample_pred_binary)
    test_acc = accuracy_score(y_test, y_pred_binary)
    print(f"Train accuracy is {train_acc:0.3f} and Test accuracy is {test_acc:0.3f}", '\n')
    make_classification_report_given_treshold(y_test, y_pred, 0.5)
    print()
    # y_pred are the predicted probabilities for the positive class (1)
    print(f'Average train precision score is {average_precision_score(y_train, in_sample_pred):0.3f}')
    print(f"Average test precision score is {average_precision_score(y_test, y_pred):0.3f}", '\n')
    

    if ret_probs:
        return y_pred


    return y_pred_binary



def perform_randomized_grid_search(X_train, y_train, algorithm, param_distributions, n_iter=100, cv=5, scoring='accuracy',\
                                   random_state=42, debug=False):
    """
    Perform a randomized grid search to find the best hyperparameters for a given algorithm.

    Parameters:
    algorithm (estimator): The machine learning algorithm to optimize.
    param_distributions (dict): The parameter distributions to sample from during the search.
    n_iter (int): Number of parameter settings that are sampled. Defaults to 100.
    cv (int): Number of cross-validation folds. Defaults to 5.
    scoring (str): The metric to optimize. Defaults to 'accuracy'.
    random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
    best_estimator (estimator): The best estimator found by the search.
    best_score (float): The best score achieved by the best estimator.
    best_params (dict): The best hyperparameters found by the search.

    Example:
    param_distributions = {
     'n_estimators': [100, 200, 300, 400, 500],
     'max_features': ['auto', 'sqrt', 'log2'],
     'max_depth': [None, 10, 20, 30, 40, 50],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4]
        }
    
    best_estimator, best_score, best_params = perform_randomized_grid_search(X_train, y_train, RandomForestClassifier(), param_distributions)
    
    print("Best Estimator:", best_estimator)
    print("Best Score:", best_score)
    print("Best Parameters:", best_params)

    Best Estimator: RandomForestClassifier(max_depth=40, max_features='log2', n_estimators=200)
    Best Score: 0.9926412487912696
    Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 40}
    """
    # Initialize the RandomizedSearchCV object
    randomized_search = RandomizedSearchCV(
        estimator=algorithm,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=2

    )

    # Fit the randomized search to the data
    randomized_search.fit(X_train, y_train)

    # Get the best estimator, score, and parameters
    best_estimator = randomized_search.best_estimator_
    best_score = randomized_search.best_score_
    best_params = randomized_search.best_params_

    if debug :
        debug_results = pd.DataFrame(randomized_search.cv_results_)

        return debug_results

    return best_estimator, best_score, best_params