import pandas as pd

def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0 / 9.0

def combine_data(first_data=None, second_data=None, first_data_csv='gottenDiagnoses.csv', has_disease_column='has_sepsis',
                 second_data_csv='ssir.csv', title_column='long_title', subject_id_column='subject_id',
                 output_information_df='sepsis_info_df.csv', disease_str='sepsis', translate_columns=None,
                 translate_functions=None, value_column='valueuom', log_stats=True, output_file=None):
    """
    Combines diagnoses and SSIR data, translates specified columns using given functions,
    and logs statistics about sepsis patients.
    Also, it can combine any other CSV data frames as long as both of them have specified columns.
    The first should have title_column and subject_id, the second data_frame should have at least the subject_id column.
    The output file will have a boolean value about long_title containing disease_str value.

    Args:
        first_data (pd.DataFrame, optional): DataFrame containing diagnoses data. Default is None.
        first_data_csv (str, optional): Path to the CSV file containing diagnoses data. Default is 'gottenDiagnoses.csv'.
        has_disease_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
        second_data (pd.DataFrame, optional): DataFrame containing SSIR data. Default is None.
        second_data_csv (str, optional): Path to the CSV file containing SSIR data. Default is 'ssir.csv'.
        title_column (str): Column name for diagnosis titles. Default is 'long_title'.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        output_information_df (str): Path to the output CSV file for sepsis information. Default is 'sepsis_info_df.csv'.
        disease_str (str): String to identify sepsis-related diagnoses. Default is 'sepsis'.
        translate_columns (dict, optional): Dictionary where keys are columns to translate from and values are columns to translate to. Default is None.
        translate_functions (dict, optional): Dictionary where keys are columns to translate from and values are functions to use for translation. Default is None.
        value_column (str): Column name to be excluded from the merged data. Default is 'valueuom'.
        log_stats (bool): Whether to log statistics about sepsis patients. Default is True.
        output_file (str, optional): Path to the output CSV file for combined data. Default is None.

    Returns:
        pd.DataFrame: The combined and processed data.
    """
    if first_data is None and first_data_csv is not None:
        diagnoses = pd.read_csv(first_data_csv)
    elif first_data is not None:
        diagnoses = first_data
    else:
        raise ValueError("Either first_data or first_data_csv must be provided.")

    if second_data is None and second_data_csv is not None:
        ssir = pd.read_csv(second_data_csv)
    elif second_data is not None:
        ssir = second_data
    else:
        raise ValueError("Either second_data or second_data_csv must be provided.")

    diagnoses[has_disease_column] = diagnoses[title_column].str.contains(disease_str, case=False, na=False)
    sepsis_info_df = diagnoses.groupby(subject_id_column)[has_disease_column].any().reset_index()
    sepsis_info_df.to_csv(output_information_df, index=False)
    merged_df = pd.merge(ssir, sepsis_info_df, on=subject_id_column, how='left')
    merged_df.drop(columns=[col for col in merged_df.columns if value_column in col], inplace=True)

    if translate_columns and translate_functions:
        for from_col, to_col in translate_columns.items():
            if from_col in translate_functions:
                translate_function = translate_functions[from_col]
                merged_df[to_col] = merged_df.apply(
                    lambda row: translate_function(row[from_col]) if pd.notnull(row[from_col]) else row[to_col],
                    axis=1
                )
                merged_df.drop(columns=[from_col], inplace=True)

    if output_file is not None:
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
        print(f'Unique patients with sepsis predictions: {count_with_sepsis}')
        print(f'Unique patients without sepsis predictions: {count_without_sepsis}')
        print(f'Patients with both sepsis and no sepsis records: {count_ambiguous_sepsis}')
        print(f'Total unique patients: {len(grouped_sepsis)}')

    return merged_df

# df1 = pd.read_csv('gottenDiagnoses.csv')
# df2 = pd.read_csv('ssir.csv')
# combined_df = combine_data(first_data=df1, second_data=df2, translate_columns=translate_columns, translate_functions=translate_functions)
# combined_df.to_csv('combined_data.csv', index=False)