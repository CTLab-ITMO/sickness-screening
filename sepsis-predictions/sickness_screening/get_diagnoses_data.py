import pandas as pd


def get_diagnoses_data(patient_diagnoses_df=None, all_diagnoses_df=None, patient_diagnoses_csv='diagnoses.csv',
                       all_diagnoses_csv='d_icd_diagnoses.csv', diagnoses_code_column='icd_code',
                       title_column='long_title', subject_id_column='subject_id', output_file_csv=None):
    """
    Extracts and merges patient diagnosis data with a general diagnosis reference and saves the result to a CSV file if specified.

    Args:
        patient_diagnoses_df (pd.DataFrame, optional): DataFrame with patient diagnoses. Default is None.
        all_diagnoses_df (pd.DataFrame, optional): DataFrame with the general diagnosis reference. Default is None.
        patient_diagnoses_csv (str, optional): Path to the CSV file with patient diagnoses. Default is 'diagnoses.csv'.
        all_diagnoses_csv (str, optional): Path to the CSV file with the general diagnosis reference. Default is 'd_icd_diagnoses.csv'.
        diagnoses_code_column (str): Name of the column with diagnosis codes. Default is 'icd_code'.
        title_column (str): Name of the column with diagnosis descriptions. Default is 'long_title'.
        subject_id_column (str): Name of the column with patient identifiers. Default is 'subject_id'.
        output_file_csv (str, optional): Path to the CSV file to save the result. Default is None.

    Returns:
        pd.DataFrame: The aggregated diagnosis data.
    """
    if patient_diagnoses_df is None and patient_diagnoses_csv is not None:
        df_diagnos = pd.read_csv(patient_diagnoses_csv)
    elif patient_diagnoses_df is not None:
        df_diagnos = patient_diagnoses_df
    else:
        raise ValueError("Either patient_diagnoses_df or patient_diagnoses_csv must be provided.")

    if all_diagnoses_df is None and all_diagnoses_csv is not None:
        df_d_diagnos = pd.read_csv(all_diagnoses_csv)
    elif all_diagnoses_df is not None:
        df_d_diagnos = all_diagnoses_df
    else:
        raise ValueError("Either all_diagnoses_df or all_diagnoses_csv must be provided.")

    diagnos_results = df_diagnos[df_diagnos[diagnoses_code_column].isin(df_d_diagnos[diagnoses_code_column])]
    diagnos_results = diagnos_results.merge(df_d_diagnos[[diagnoses_code_column, title_column]],
                                            on=diagnoses_code_column)
    diagnos_results = diagnos_results[[subject_id_column, title_column]]
    if output_file_csv is not None:
        diagnos_results.to_csv(output_file_csv, index=False)

    return diagnos_results

# df_patient_diagnoses = pd.read_csv('diagnoses.csv')
# df_all_diagnoses = pd.read_csv('d_icd_diagnoses.csv')
# result_df = get_diagnoses_data(patient_diagnoses_df=df_patient_diagnoses, all_diagnoses_df=df_all_diagnoses)
# result_df.to_csv('gottenDiagnoses.csv', index=False)
