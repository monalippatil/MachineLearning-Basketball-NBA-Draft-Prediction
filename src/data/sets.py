import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defining a function to arrgange date values properly in the 'ht' height feature
def arranging_date(datapoint):
    """
    Arrange date values properly in the 'ht' height feature

    Parameters
    ----------
    datapoint : object
        Input datapoint value

    Returns
    -------
    datapoint : object
        Arranged datapoint value of 'ht' height categorical feature
    """
    if datapoint == 'Jun-00':
        return '00-Jun'
    elif datapoint == 'Jul-00':
        return '00-Jul'
    elif datapoint == 'Apr-00':
        return '00-Apr'
    else:
        return datapoint


# Defining a function to modifying date values to relevant height  
def replacing_ft(datapoint):
    """
    Replace date values to relevant height

    Parameters
    ----------
    datapoint : object
        Input datapoint value

    Returns
    -------
    datapoint : object
        Modified the date values to relevant height  
    """
    if datapoint == 'Jun':
        return '06'
    elif datapoint == 'Jul':
        return '07'
    elif datapoint == 'Apr':
        return '04'
    elif datapoint == 'May':
        return '05'
    else:
        return datapoint
  
    
# Defining a function to identify missing values in the dataset
def displaying_null_values(df):
    """
    Identify if any null values in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    """
    
    print('Total number of features with null values:', df.isnull().sum()[df.isnull().sum() > 0].count())

    # Collecting features with null values
    df_features_with_null = df.columns[df.isnull().any()]

    # Displaying features names with null values
    print('\nFeatures with null values:\n', df_features_with_null)

    # Displaying features with null values and their null value counts
    print('\nFeatures with null values and their counts:')
    for feature in df_features_with_null:
        null_count = df[feature].isnull().sum()
        print(f'{feature}: {null_count}')


# Defining a function to impute the missing values with mean for numerical features
def imputing_missing_with_mean(df_data):
    """
    Impute the missing values with mean for numerical features of the dataset and retured the replaced dataset

    Parameters
    ----------
    df_data : pd.DataFrame
        Input dataframe

    Returns
    -------
    df_data : pd.DataFrame
        Altered dataframe with mean values replacing any missing data in the numerical features
    """
    
    numerical_features = df_data.select_dtypes(include='float')
    # checking and replacing missing value with the mean
    for feature in numerical_features.columns:
        df_data[feature].fillna(numerical_features[feature].mean(), inplace=True)
    
    return df_data


# Defining a function to impute the missing values with mode for categorical features
def imputing_missing_with_mode(df_data):
    """
    Impute the missing values with mode for categorical features of the dataset and retured the replaced dataset

    Parameters
    ----------
    df_data : pd.DataFrame
        Input dataframe

    Returns
    -------
    df_data : pd.DataFrame
        Altered dataframe with mode values replacing any missing data in the categorical features
    """
    
    categorical_features = df_data.select_dtypes(include='object')
    # checking and replacing missing value with the mode
    for feature in categorical_features.columns:
        df_data[feature].fillna(categorical_features[feature].mode()[0], inplace=True)
    
    return df_data


# Defining a function to access the response variable classess to verify imbalance
def assessing_if_imbalance_dataset(df, response_variable):
    """
    Accessing the classes of response variable to verify if datdaset is imbalance
     
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    response_variable: string
        Name of the target variable

    Returns
    -------
    """
    
    # Examining the number of classes within the target variable
    print('\nNumber of classes within the target variable:\n', df[response_variable].value_counts())

    # Generating a pie chart to visualize the distribution of classes in the target variable
    plt.pie(df[response_variable].value_counts(), labels=df[response_variable].unique(), autopct='%1.1f%%', colors=['Orange', 'Green'])
    plt.title('Distribution of Target Classes')
    plt.show()


# Defining a function to extract target variable from the dataframe
def pop_target_variable(df, target_variable):
    """
    Extract target variable from the dataframe and return the features dataset and target variable seperated

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_variable : string
        Name of the target variable

    Returns
    -------
    df_copy  : pd.DataFrame
        Subsetted Pandas dataframe containing all features
    response_variable : pd.Series
        Subsetted Pandas dataframe containing the target variable
    """

    # Creating a copy of the input dataset and separating target variable from the features
    df_copy = df.copy()
    response_variable = df_copy.pop(target_variable)

    return df_copy, response_variable


# Defining a function to address imbalanced target classes using the SMOTE oversampling technique
def oversampling_with_smote(predictor_features, response_variable):
    """
    Apply SMOTE oversampling method to equalize classes of the target variable and return the resampled datasets
   
    Parameters
    ----------
    predictor_features : pd.DataFrame
        Input dataframe with all predictors
    target_variable : pd.Series
        Target variable

    Returns
    -------
    X_resampled  : pd.DataFrame
        Oversampled Pandas dataframe containing all features
    y_resampled : pd.Series
        Oversampled Pandas dataframe containing the target variable with equal classes
    """
    
    # Importing SMOTE library to perform oversampling
    from imblearn.over_sampling import SMOTE

    # Instantiating a SMOTE object to apply its oversampling technique to balance the dataset
    smote_oversmapling = SMOTE(random_state=19)

    # Creating synthetic samples of the minority class to balance the class distribution
    X_resampled, y_resampled = smote_oversmapling.fit_resample(predictor_features, response_variable)

    print('\nVerifying the number of claases within the target variable post oversmapling\n')
    # Get the value counts for each class
    class_counts = np.bincount(y_resampled)

    # Print the class value counts
    print("Class 0 count:", class_counts[0])
    print("Class 1 count:", class_counts[1])
    
    return X_resampled, y_resampled


# Defining a function to randomly partition the dataset into training and validation sets
def train_validation_split(predictors_features, target_variable, validation_ratio=0.2):
    """
    Randomly split the dataset into training and validation sets and return the split datasets

    Parameters
    ----------
    predictors_features : pd.DataFrame
        Input dataframe
    target_variable : pd.Series
        Target variable
    test_ratio : float
        Ratio used for the validation sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    """

    # Importing train_test_split library to split the dataset
    from sklearn.model_selection import train_test_split

    # Splitting the data into training set (80%) and validation set (20%)
    X_train, X_validate, y_train, y_validate = train_test_split(predictors_features, target_variable, test_size=0.2, random_state=19)
    
    return X_train, X_validate, y_train, y_validate


# Defining a function to scale the features to ensure uniformity in features values
def features_scaling(training_features, validation_features, testing_features):
    """
    Scaling features to achieve consistency in feature values and return the scaled datasets and scaler object
    
    Parameters
    ----------
    training_features : pd.DataFrame
        Input training dataframe
    validation_features : pd.DataFrame
        Input validation dataframe
    testing_features : pd.DataFrame
        Input testing dataframe

    Returns
    -------
    X_train : pd.DataFrame
        Scaled seatures of the training dataset
    X_validate : pd.DataFrame
        Scaled seatures of the validation dataset
    X_test : pd.DataFrame
        Scaled seatures of the testing dataset
    """

    # Importing StandardScaler library to scale the predictors of all the datasets
    from sklearn.preprocessing import StandardScaler

    # Creating an instance named 'scaler' of the StandardScaler class 
    scaler = StandardScaler()

    # Applying the 'scaler' to the training dataset to adjust its scale
    scaler.fit(training_features)

    # Transforming and replacing the features data from all the sets with the 'scaler' object
    X_train = scaler.transform(training_features)
    X_validate = scaler.transform(validation_features)
    X_test = scaler.transform(testing_features)   

    return X_train, X_validate, X_test, scaler


# Defining a function to store all the processed datasets prepared for the machine learning purposes
def save_datasets(X_train=None, y_train=None, X_validate=None, y_validate=None, X_test=None, y_test=None, path='../data/processed/'):
    """
    Store all the datasets locally in 'data/processed' directory that are prepared 

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training dataset
    y_train: Numpy Array
        Target for the training dataset
    X_validate: Numpy Array
        Features for the validation dataset
    y_validate: Numpy Array
        Target for the validation dataset
    X_test: Numpy Array
        Features for the testing dataset
    y_test: Numpy Array
        Target for the testing dataset
    path : string
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """

    # Saving the datasets individually
    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_validate is not None:
      np.save(f'{path}X_validate', X_validate)
    if X_test is not None:
      np.save(f'{path}X_test', X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_validate is not None:
      np.save(f'{path}y_validate', y_validate)
    if y_test is not None:
      np.save(f'{path}y_test', y_test)


# Defining a function to load all the prepared datasets to build machine learning model
def load_datasets(path='../data/processed/'):
    """
    Load all the prepared datasets stored locally in the 'data/processed' directory

    Parameters
    ----------
    path : string
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_validate : Numpy Array
        Features for the validation set
    y_validate : Numpy Array
        Target for the validation set
    X_test : Numpy Array
        Features for the testing set
    y_test : Numpy Array
        Target for the testing set
    """

    # Importing required libaries
    import os.path
    import numpy as np
    
    # Loading all the prepared datasets in their respective datasets
    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_validate = np.load(f'{path}X_validate.npy', allow_pickle=True) if os.path.isfile(f'{path}X_validate.npy')   else None
    X_test = np.load(f'{path}X_test.npy', allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_validate = np.load(f'{path}y_validate.npy', allow_pickle=True) if os.path.isfile(f'{path}y_validate.npy')   else None
    y_test = np.load(f'{path}y_test.npy', allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_validate, y_validate, X_test, y_test