import pandas as pd


def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0 / 9.0


def combine_data(first_data='gottenDiagnoses.csv', has_disease_column='has_sepsis',
                 second_data='ssir.csv', title_column='long_title', subject_id_column='subject_id',
                 output_information_df='sepsis_info_df.csv', disease_str='sepsis', has_temp=True,
                 translate_data_from_col='Temperature Fahrenheit', translate_data_to_col='Temperature Celsius',
                 translate_function=fahrenheit_to_celsius, value_column='valueuom', log_stats=True,
                 output_file='diagnoses_and_ssir.csv'):
    """
    Combines diagnoses and SSIR data, translate temperature from Fahrenheit to Celsius (or any other value if needed),
    and log statistics about sepsis patients.
    Also, it could be possible to combine any other csv data frames as long as both of them have specified columns.
    First should have title_column and subject_id, second data_frame should have at least subject_id column.
    Output file will have boolean value about long_title containing disease_str value.
    Args:
        first_data (str): Path to the CSV file containing diagnoses data. Default is 'gottenDiagnoses.csv'.
        has_disease_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
        second_data (str): Path to the CSV file containing SSIR data. Default is 'ssir.csv'.
        title_column (str): Column name for diagnosis titles. Default is 'long_title'.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        output_information_df (str): Path to the output CSV file for sepsis information. Default is 'sepsis_info_df.csv'.
        disease_str (str): String to identify sepsis-related diagnoses. Default is 'sepsis'.
        has_temp (bool): Whether to process temperature data. Default is True.
        translate_data_from_col (str): Column name for temperature in Fahrenheit. Default is 'Temperature Fahrenheit'.
        translate_data_to_col (str): Column name for temperature in Celsius. Default is 'Temperature Celsius'.
        translate_function (function): Function to convert Fahrenheit to Celsius. Default is fahrenheit_to_celsius.
        value_column (str): Column name to be excluded from the merged data. Default is 'valueom'.
        log_stats (bool): Whether to log statistics about sepsis patients. Default is True.
        output_file (str): Path to the output CSV file for combined data. Default is 'diagnoses_and_ssir.csv'.

    Returns:
        None: Writes the combined and processed data to the specified output file.
    """
    diagnoses = pd.read_csv(first_data)
    ssir = pd.read_csv(second_data)
    diagnoses[has_disease_column] = diagnoses[title_column].str.contains(disease_str, case=False, na=False)
    sepsis_info_df = diagnoses.groupby(subject_id_column)[has_disease_column].any().reset_index()
    sepsis_info_df.to_csv(output_information_df)
    merged_df = pd.merge(ssir, sepsis_info_df, on=subject_id_column, how='left')
    merged_df.drop(columns=[col for col in merged_df.columns if value_column in col], inplace=True)
    if has_temp:
        merged_df[translate_data_to_col] = merged_df.apply(
            lambda row: translate_function(row[translate_data_from_col]) if pd.notnull(
                row[translate_data_from_col]) else
            row[translate_data_to_col],
            axis=1
        )
        merged_df.drop(columns=[translate_data_from_col], inplace=True)
    merged_df.to_csv(output_file, index=False)
    if log_stats:
        ans = sepsis_info_df[has_disease_column].sum()
        unique_patients = merged_df[[subject_id_column, has_disease_column]].drop_duplicates()
        sepsis_counts = unique_patients[has_disease_column].value_counts(normalize=False)
        count_with_sepsis = sepsis_counts.get(True, 0)
        count_without_sepsis = sepsis_counts.get(False, 0)
        grouped_sepsis = unique_patients.groupby(subject_id_column)[has_disease_column].agg(['min', 'max'])
        ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']]
        count_ambiguous_sepsis = len(ambiguous_sepsis_patients)
        print(f'Correct number of patients with sepsis: {ans}')
        print(f'Unique patients with predictions_sepsis: {count_with_sepsis}')
        print(f'Unique patients without predictions_sepsis: {count_without_sepsis}')
        print(f'Patients with both predictions_sepsis and no predictions_sepsis records: {count_ambiguous_sepsis}')
        print(f'All unique patients: {len(grouped_sepsis)}')
