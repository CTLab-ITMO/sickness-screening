import pandas as pd


def choose(compressed_df_csv='compressed.csv', has_sepsis_col='has_sepsis', subject_id_col='subject_id',
           output_file='balance_need_filling.csv'):
    """
    Selects a balanced subset of the dataset by choosing the top non-sepsis patients matching the number of sepsis patients,
    and saves the resulting dataset to a CSV file.

    This function reads a compressed dataset, separates sepsis and non-sepsis patients, selects the top non-sepsis patients
    based on the number of records to match the number of sepsis patients, and concatenates them into a final dataset.

    Args:
        compressed_df_csv (str): Path to the CSV file containing the compressed data. Default is 'compressed.csv'.
        has_sepsis_col (str): Column name indicating the presence of sepsis (True for sepsis, False for non-sepsis). Default is 'has_sepsis'.
        subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
        output_file (str): Path to the output CSV file for the balanced dataset. Default is 'balance_need_filling.csv'.

    Returns:
        None: Writes the balanced dataset to the specified output file.
    """
    df = pd.read_csv(compressed_df_csv)
    df_sepsis = df[df[has_sepsis_col] == True]
    df_no_sepsis = df[df[has_sepsis_col] == False]
    count_no_sepsis = df_no_sepsis[subject_id_col].value_counts()
    sorted_no_sepsis = count_no_sepsis.sort_values(ascending=False)
    num_sepsis_patients = df_sepsis[subject_id_col].nunique()
    top_no_sepsis = sorted_no_sepsis.head(num_sepsis_patients).index.tolist()
    selected_no_sepsis = df_no_sepsis[df_no_sepsis[subject_id_col].isin(top_no_sepsis)]
    final_dataset = pd.concat([df_sepsis, selected_no_sepsis])
    final_dataset.to_csv(output_file)
