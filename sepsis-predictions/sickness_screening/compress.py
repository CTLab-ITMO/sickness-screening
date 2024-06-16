import pandas as pd


def compress(df_to_compress=None, subject_id_col='subject_id', output_csv=None):
    """
    Compresses the dataset by forward-filling and backward-filling missing values within each group identified by subject IDs,
    and removes duplicate rows. The resulting dataset is saved to a CSV file.

    Args:
        df_to_compress (pd.DataFrame, optional): DataFrame containing the data to compress. Default is None.
        subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
        output_csv (str, optional): Path to the output CSV file for the compressed data. Default is 'compressed.csv'.

    Returns:
        pd.DataFrame: The compressed data.
    """
    if df_to_compress is None:
        raise ValueError("df_to_compress must be provided.")

    df = df_to_compress.groupby(subject_id_col, group_keys=False).apply(
        lambda group: group.ffill().bfill().infer_objects(copy=False)
    ).reset_index(drop=True)
    df = df.drop_duplicates()

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df