import pandas as pd


def compress(df_to_compress='diagnoses_and_ssir_and_blood_and_chartevents.csv', subject_id_col='subject_id',
             output_csv='compressed.csv'):
    """
    Compresses the dataset by forward-filling and backward-filling missing values within each group identified by subject IDs,
    and removes duplicate rows. The resulting dataset is saved to a CSV file.

    Args:
        df_to_compress (str): Path to the CSV file containing the data to compress. Default is 'diagnoses_and_ssir_and_blood_and_chartevents.csv'.
        subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
        output_csv (str): Path to the output CSV file for the compressed data. Default is 'compressed.csv'.

    Returns:
        None: Writes the compressed data to the specified output file.
    """
    df = pd.read_csv(df_to_compress)
    df = df.groupby(subject_id_col).apply(lambda group: group.bfill().ffill()).reset_index(drop=True)
    df = df.drop_duplicates()
    df.to_csv(output_csv)

