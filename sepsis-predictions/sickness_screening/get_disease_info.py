import pandas as pd


def get_diseas_info(diagnoses_csv='gottenDiagnoses.csv', title_column='long_title', diseas_str='sepsis',
                    diseas_column='has_sepsis', subject_id_column='subject_id', log_stats=True,
                    output_csv='sepsis_info.csv'):
    """
    Extracts sepsis information from diagnoses data and logs statistics.

    This function reads diagnoses data from a CSV file, identifies sepsis-related diagnoses by checking if the
    'long_title' contains the specified 'diseas_str', and logs statistics about the total number of patients
    and those with sepsis. The results are saved to 'output_csv'.

    Args:
        diagnoses_csv (str): Path to the CSV file containing diagnoses data. Default is 'gottenDiagnoses.csv'.
        title_column (str): Column name for diagnosis titles. Default is 'long_title'.
        diseas_str (str): String to identify sepsis-related diagnoses. Default is 'sepsis'.
        diseas_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        log_stats (bool): Whether to log statistics about sepsis patients. Default is True.

    Returns:
        None : writes information to 'output_csv'
    """
    diagnoses = pd.read_csv(diagnoses_csv)
    diagnoses[diseas_column] = diagnoses[title_column].str.contains(diseas_str, case=False, na=False)
    sepsis_info_df = diagnoses[[subject_id_column, diseas_column]].drop_duplicates()
    sum = 0
    length = 0
    for index, row in sepsis_info_df.iterrows():
        length += 1
        if (row[diseas_column]):
            sum += 1
    if log_stats:
        print(f'Всего пациентов: {length}')
        print(f'Всего пациентов с сепсисом {sum}')
    sepsis_info_df.to_csv(output_csv, index=False)
