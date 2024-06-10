import pandas as pd

def choose(compressed_df=None, compressed_df_csv=None, has_disease_col='has_sepsis', subject_id_col='subject_id',
           output_file=None):
    """
    Selects a balanced subset of the dataset by choosing the top non-sepsis patients matching the number of sepsis patients,
    and optionally saves the resulting dataset to a CSV file.

    This function reads a compressed dataset, separates sepsis and non-sepsis patients, selects the top non-sepsis patients
    based on the number of records to match the number of sepsis patients, and concatenates them into a final dataset.

    Args:
        compressed_df (pd.DataFrame, optional): DataFrame containing the compressed data. Default is None.
        compressed_df_csv (str, optional): Path to the CSV file containing the compressed data. Default is None.
        has_disease_col (str): Column name indicating the presence of sepsis (True for sepsis, False for non-sepsis). Default is 'has_sepsis'.
        subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
        output_file (str, optional): Path to the output CSV file for the balanced dataset. Default is None.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    if compressed_df is None and compressed_df_csv is not None:
        df = pd.read_csv(compressed_df_csv)
    elif compressed_df is not None:
        df = compressed_df
    else:
        raise ValueError("Either compressed_df or compressed_df_csv must be provided.")

    df_sepsis = df[df[has_disease_col] == True]
    df_no_sepsis = df[df[has_disease_col] == False]
    count_no_sepsis = df_no_sepsis[subject_id_col].value_counts()
    sorted_no_sepsis = count_no_sepsis.sort_values(ascending=False)
    num_sepsis_patients = df_sepsis[subject_id_col].nunique()
    top_no_sepsis = sorted_no_sepsis.head(num_sepsis_patients).index.tolist()
    selected_no_sepsis = df_no_sepsis[df_no_sepsis[subject_id_col].isin(top_no_sepsis)]
    final_dataset = pd.concat([df_sepsis, selected_no_sepsis])
    if output_file is not None:
        final_dataset.to_csv(output_file, index=False)

    return final_dataset


# df = pd.read_csv('compressed.csv')
# balanced_df = choose(compressed_df=df, has_disease_col='has_sepsis', subject_id_col='subject_id')
# balanced_df.to_csv('balanced_data.csv', index=False)
