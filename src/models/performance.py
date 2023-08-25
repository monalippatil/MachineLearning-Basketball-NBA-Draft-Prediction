import pandas as pd
import numpy as np

# Defining a function to diplay the AUROC performance score for the given dataset
def display_performance_score(y_variable_predicts, y_variable_actuals, dataset_name=None):
    """
    Show the AUROC performance score and visual representation of ROC curve for the given dataset

    Parameters
    ----------
    y_variable_predicts : Numpy Array
        Predicted target variable's class values
    y_variable_actuals : Numpy Array
        Actual target variable's class values
    dataset_name : str
        Name of the dataset to be displayed

    Returns
    -------
    """

    # Importing roc_auc_score, roc_curve and matplotlib libraries required for visual representation
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    # Displaying the AUROC performance scores for the given dataset
    print(f"{dataset_name} AUROC score:", roc_auc_score(y_variable_actuals, y_variable_predicts))

    auroc_score = roc_auc_score(y_variable_actuals, y_variable_predicts)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_variable_actuals, y_variable_predicts)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


# Defining a function to predict and evaluate the AUROC performance score for the given dataset
def evaluate_classifier_dataset(classifier_mode, X_feature, y_variable, dataset_name=''):
    """
    Save the predictions from a trained model on a given dataset and display its AUROC peerformance scores

    Parameters
    ----------
    classifier_mode: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    X_feature : Numpy Array
        Features of the specified dataset
    y_variable : Numpy Array
        Target variable of the specified dataset
    dataset_name : str
        Name of the set to be displayed

    Returns
    -------
    """

    # Perfoming prediction from the trained model on a given dataset
    y_variable_predicts = classifier_mode.predict(X_feature)

    # Invoking the function to display the performance score of the given dataset
    display_performance_score(y_variable_predicts, y_variable_actuals=y_variable, dataset_name=dataset_name)


# Defining a function to train the model, predict and evaluate the result for training dataset
def train_evaluate_classifier(classifier_mode, X_train, y_train):
    """
    Train a classification model, predict the target variable classses and evaluate the performance for the training dataset and retrun the trained model

    Parameters
    ----------
    classifier_mode: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training dataset
    y_train : Numpy Array
        Target for the training dataset

    Returns
    classifier_mode : sklearn.base.BaseEstimator
        Trained model
    -------
    """  
    
    # Training the classification model using the features of the training dataset
    classifier_mode.fit(X_train, y_train)
    
    # Invoking the function to predict and evaluate the performance for the given dataset
    evaluate_classifier_dataset(classifier_mode, X_train, y_train, dataset_name='Training')
    
    return classifier_mode	