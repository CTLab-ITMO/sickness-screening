import pandas as pd


def get_disease_info(diagnoses_df=None, diagnoses_csv='gottenDiagnoses.csv', title_column='long_title',
                     disease_str='sepsis', disease_column='has_disease', subject_id_column='subject_id',
                     log_stats=True, output_csv=None):
    """
    Extracts disease information from diagnoses data and logs statistics.

    This function reads diagnoses data from a DataFrame or a CSV file, identifies disease-related diagnoses by checking if the
    'long_title' contains the specified 'disease_str', and logs statistics about the total number of patients
    and those with the disease. The results are optionally saved to 'output_csv'.

    Args:
        diagnoses_df (pd.DataFrame, optional): DataFrame containing diagnoses data. Default is None.
        diagnoses_csv (str, optional): Path to the CSV file containing diagnoses data. Default is 'gottenDiagnoses.csv'.
        title_column (str): Column name for diagnosis titles. Default is 'long_title'.
        disease_str (str): String to identify disease-related diagnoses. Default is 'sepsis'.
        disease_column (str): Column name to indicate disease presence. Default is 'has_disease'.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        log_stats (bool): Whether to log statistics about disease patients. Default is True.
        output_csv (str, optional): Path to the output CSV file for the disease information. Default is None.

    Returns:
        pd.DataFrame: The disease information data.
    """
    if diagnoses_df is None and diagnoses_csv is not None:
        diagnoses = pd.read_csv(diagnoses_csv)
    elif diagnoses_df is not None:
        diagnoses = diagnoses_df
    else:
        raise ValueError("Either diagnoses_df or diagnoses_csv must be provided.")

    diagnoses[disease_column] = diagnoses[title_column].str.contains(disease_str, case=False, na=False)
    disease_info_df = diagnoses[[subject_id_column, disease_column]].drop_duplicates()
    disease_info_resolved = disease_info_df.groupby(subject_id_column)[disease_column].any().reset_index()

    if log_stats:
        total_patients = len(disease_info_resolved)
        disease_patients = disease_info_resolved[disease_column].sum()
        print(f'Total patients: {total_patients}')
        print(f'Total patients with disease: {disease_patients}')

    if output_csv is not None:
        disease_info_resolved.to_csv(output_csv, index=False)

    return disease_info_resolved

# df_diagnoses = pd.read_csv('gottenDiagnoses.csv')
# disease_info = get_disease_info(diagnoses_df=df_diagnoses, title_column='long_title', disease_str='sepsis', subject_id_column='subject_id')
# disease_info.to_csv('disease_info.csv', index=False)
