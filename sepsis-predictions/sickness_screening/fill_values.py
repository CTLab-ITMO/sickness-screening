import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def fill_values(balanced_df=None, balanced_csv='balance_need_filling.csv', strategy='most_frequent', output_csv=None):
    """
    Fills missing values in the dataset using the specified imputation strategy and saves the resulting dataset to a CSV file if specified.

    This function reads a dataset, replaces specified placeholders for missing values with NaN, imputes missing values
    using the specified strategy, and saves the resulting dataset to a CSV file. If you are using a different strategy you should be sure
    that all columns in 'balanced_csv' are numeric.

    Args:
        balanced_df (pd.DataFrame, optional): DataFrame containing the data to be imputed. Default is None.
        balanced_csv (str, optional): Path to the CSV file containing the data to be imputed. Default is 'balance_need_filling.csv'.
        strategy (str): Imputation strategy to use. Options are 'mean', 'median', 'most_frequent', and 'constant'. Default is 'most_frequent'.
        output_csv (str, optional): Path to the output CSV file for the imputed dataset. Default is None.

    Returns:
        pd.DataFrame: The imputed dataset.
    """
    if balanced_df is None and balanced_csv is not None:
        df = pd.read_csv(balanced_csv)
    elif balanced_df is not None:
        df = balanced_df
    else:
        raise ValueError("Either balanced_df or balanced_csv must be provided.")
    df.replace('___', np.nan, inplace=True)
    df.replace('-', np.nan, inplace=True)
    imputer_mode = SimpleImputer(strategy=strategy)
    df_filled_mode = imputer_mode.fit_transform(df)
    df_filled_mode = pd.DataFrame(df_filled_mode, columns=df.columns)
    if output_csv is not None:
        df_filled_mode.to_csv(output_csv, index=False)

    return df_filled_mode

# df = pd.read_csv('balance_need_filling.csv')
# filled_df = fill_values(balanced_df=df, strategy='most_frequent')
# filled_df.to_csv('filled_data.csv', index=False)
