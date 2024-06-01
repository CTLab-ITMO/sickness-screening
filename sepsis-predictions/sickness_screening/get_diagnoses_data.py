import pandas as pd
def get_diagnoses_data(patient_diagnoses_csv='diagnoses.csv', all_diagnoses_csv='d_icd_diagnoses.csv',
                       diagnoses_code_column='icd_code', title_column='long_title', subject_id_column='subject_id',
                       output_file_csv='gottenDiagnoses.csv'):
    """
    Extracts and merges patient diagnosis data with a general diagnosis reference and saves the result to a CSV file.

    Args:
        patient_diagnoses_csv (str): Path to the CSV file with patient diagnoses. Default is 'diagnoses.csv'.
        all_diagnoses_csv (str): Path to the CSV file with the general diagnosis reference. Default is 'd_icd_diagnoses.csv'.
        diagnoses_code_column (str): Name of the column with diagnosis codes. Default is 'icd_code'.
        title_column (str): Name of the column with diagnosis descriptions. Default is 'long_title'.
        subject_id_column (str): Name of the column with patient identifiers. Default is 'subject_id'.
        output_file_csv (str): Path to the CSV file to save the result. Default is 'gottenDiagnoses.csv'.

    Returns:
        None: Writes the aggregated diagnosis file to output_file_csv.
    """
    df_d_diagnos = pd.read_csv(all_diagnoses_csv)
    df_diagnos = pd.read_csv(patient_diagnoses_csv)
    diagnos_results = df_diagnos[df_diagnos[diagnoses_code_column].isin(df_d_diagnos[diagnoses_code_column])]
    diagnos_results = diagnos_results.merge(df_d_diagnos[[diagnoses_code_column, title_column]],
                                            on=diagnoses_code_column)
    diagnos_results = diagnos_results[[subject_id_column, title_column]]
    diagnos_results.to_csv(output_file_csv, index=False)


get_diagnoses_data()