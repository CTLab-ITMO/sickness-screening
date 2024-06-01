import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
def fill_values(balanced_csv='balance_need_filling.csv', strategy='most_frequent', output_csv='balance_filled_mode.csv'):
    """
    Fills missing values in the dataset using the specified imputation strategy and saves the resulting dataset to a CSV file.

    This function reads a dataset, replaces specified placeholders for missing values with NaN, imputes missing values
    using the specified strategy, and saves the resulting dataset to a CSV file. If you are using different strategy you should be sure,
    that all columns in 'balanced_csv' are numeric.

    Args:
        balanced_csv (str): Path to the CSV file containing the data to be imputed. Default is 'balance_need_filling.csv'.
        strategy (str): Imputation strategy to use. Options are 'mean', 'median', 'most_frequent', and 'constant'. Default is 'most_frequent'.
        output_csv (str): Path to the output CSV file for the imputed dataset. Default is 'balance_filled_mode.csv'.
    Returns:
        None: Writes the imputed dataset to the specified output file.
    """
    df = pd.read_csv(balanced_csv)
    df.replace('___', np.nan, inplace=True)
    df.replace('-', np.nan, inplace=True)
    imputer_mode = SimpleImputer(strategy=strategy)
    df_filled_mode = imputer_mode.fit_transform(df)
    df_filled_mode = pd.DataFrame(df_filled_mode, columns=df.columns)
    df_filled_mode.to_csv(output_csv)