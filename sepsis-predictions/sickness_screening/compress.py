import pandas as pd


def compress(df_to_compress=None, df_to_compress_csv='diagnoses_and_ssir_and_blood_and_chartevents.csv',
             subject_id_col='subject_id',
             output_csv=None):
    """
    Compresses the dataset by forward-filling and backward-filling missing values within each group identified by subject IDs,
    and removes duplicate rows. The resulting dataset is saved to a CSV file if specified.

    Args:
        df_to_compress (pd.DataFrame, optional): DataFrame containing the data to compress. Default is None.
        df_to_compress_csv (str, optional): Path to the CSV file containing the data to compress. Default is 'diagnoses_and_ssir_and_blood_and_chartevents.csv'.
        subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
        output_csv (str, optional): Path to the output CSV file for the compressed data. Default is None.

    Returns:
        pd.DataFrame: The compressed data.
    """
    if df_to_compress is None and df_to_compress_csv is not None:
        df = pd.read_csv(df_to_compress_csv)
    elif df_to_compress is not None:
        df = df_to_compress
    else:
        raise ValueError("Either df_to_compress or df_to_compress_csv must be provided.")
    df = df.groupby(subject_id_col).apply(lambda group: group.bfill().ffill()).reset_index(drop=True)
    df = df.drop_duplicates()
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df
# df = pd.read_csv('diagnoses_and_ssir_and_blood_and_chartevents.csv')
# compressed_df = compress(df_to_compress=df, subject_id_col='subject_id')
# compressed_df.to_csv('compressed_data.csv', index=False)
