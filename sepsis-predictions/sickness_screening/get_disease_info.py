import pandas as pd


def get_diseas_info(diagnoses_df=None, diagnoses_csv='gottenDiagnoses.csv', title_column='long_title',
                    diseas_str='sepsis',
                    diseas_column='has_sepsis', subject_id_column='subject_id', log_stats=True, output_csv=None):
    """
    Extracts sepsis information from diagnoses data and logs statistics.

    This function reads diagnoses data from a DataFrame or a CSV file, identifies sepsis-related diagnoses by checking if the
    'long_title' contains the specified 'diseas_str', and logs statistics about the total number of patients
    and those with sepsis. The results are optionally saved to 'output_csv'.

    Args:
        diagnoses_df (pd.DataFrame, optional): DataFrame containing diagnoses data. Default is None.
        diagnoses_csv (str, optional): Path to the CSV file containing diagnoses data. Default is 'gottenDiagnoses.csv'.
        title_column (str): Column name for diagnosis titles. Default is 'long_title'.
        diseas_str (str): String to identify sepsis-related diagnoses. Default is 'sepsis'.
        diseas_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        log_stats (bool): Whether to log statistics about sepsis patients. Default is True.
        output_csv (str, optional): Path to the output CSV file for the sepsis information. Default is None.

    Returns:
        pd.DataFrame: The sepsis information data.
    """
    if diagnoses_df is None and diagnoses_csv is not None:
        diagnoses = pd.read_csv(diagnoses_csv)
    elif diagnoses_df is not None:
        diagnoses = diagnoses_df
    else:
        raise ValueError("Either diagnoses_df or diagnoses_csv must be provided.")

    diagnoses[diseas_column] = diagnoses[title_column].str.contains(diseas_str, case=False, na=False)
    sepsis_info_df = diagnoses[[subject_id_column, diseas_column]].drop_duplicates()

    if log_stats:
        total_patients = len(sepsis_info_df)
        sepsis_patients = sepsis_info_df[diseas_column].sum()
        print(f'Всего пациентов: {total_patients}')
        print(f'Всего пациентов с сепсисом: {sepsis_patients}')

    if output_csv is not None:
        sepsis_info_df.to_csv(output_csv, index=False)

    return sepsis_info_df

# df_diagnoses = pd.read_csv('gottenDiagnoses.csv')
# sepsis_info = get_diseas_info(diagnoses_df=df_diagnoses, title_column='long_title', diseas_str='sepsis', subject_id_column='subject_id')
# sepsis_info.to_csv('sepsis_info.csv', index=False)
