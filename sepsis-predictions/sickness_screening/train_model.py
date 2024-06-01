import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


def convert_values(x):
    if x == '<0.1':
        return 0
    else:
        return x


def train_model(df_to_train_csv='balance_filled_mode_v2.csv',
                categorical_col=None, columns_to_train_on=None,
                model=RandomForestClassifier(), single_cat_column='White Blood Cells',
                convert_values=convert_values, has_disease_col='has_sepsis',
                subject_id_col='subject_id', valueuom_col='valueuom',
                scaler=MinMaxScaler(), random_state=42, test_size=0.2):
    """
        Trains a machine learning model on the given dataset and evaluates its performance.

        This function reads the dataset from a CSV file, processes the data, trains a model, and evaluates its performance.
        It handles categorical columns, scales numeric features, and splits the data into training and test sets.

        Args:
            df_to_train_csv (str): Path to the CSV file containing the data to train on. Default is 'balance_filled_mode_v2.csv'.
            categorical_col (list of str, optional): List of categorical column names to be one-hot encoded. Default is None, which uses ['Large Platelets'].
            columns_to_train_on (list of str, optional): List of numeric column names to train on. Default is None, which uses ['Amylase'].
            model (any): The machine learning model to train. Default is RandomForestClassifier(). Model should have .fit and .predict methods.
            single_cat_column (str): Column name for the single categorical column to convert values. Default is 'White Blood Cells'.
            convert_values (function): Function to convert specific values in the single_cat_column. Default is convert_values.
            has_disease_col (str): Column name indicating the presence of the disease. Default is 'has_sepsis'.
            subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
            valueuom_col (str): Column name for units of measurement. Default is 'valueuom'.
            scaler (any): The scaler to use for numeric feature scaling. Default is MinMaxScaler(). Scaler should have an .fit_transoform method.
            random_state (int): Random state for reproducibility. Default is 42.
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.

        Returns:
            object: The trained machine learning model.
        """
    if columns_to_train_on is None:
        columns_to_train_on = ['Amylase']
    if categorical_col is None:
        categorical_col = ['Large Platelets']
    df = pd.read_csv(df_to_train_csv)
    df.replace('___', np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    columns_to_train_on = []
    category_columns = categorical_col
    df[single_cat_column] = df[single_cat_column].apply(convert_values)
    all_columns = columns_to_train_on.copy()
    all_columns.append(has_disease_col)
    df = pd.get_dummies(df, columns=category_columns)
    features = df[columns_to_train_on]
    df[columns_to_train_on] = scaler.fit_transform(features)
    numeric_and_one_hot_columns = columns_to_train_on + [col for col in df.columns if col.startswith(
        tuple(category_columns)) and valueuom_col not in col]
    unique_ids = df[subject_id_col].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_df = df[df[subject_id_col].isin(train_ids)]
    test_df = df[df[subject_id_col].isin(test_ids)]
    X_train = train_df[numeric_and_one_hot_columns]
    y_train = train_df[has_disease_col]
    X_test = test_df[numeric_and_one_hot_columns]
    y_test = test_df[has_disease_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model
