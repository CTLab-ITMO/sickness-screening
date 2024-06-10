import pandas as pd


def merge_and_get_data(analyzes_names=None, merge_with_df=None, merge_with='diagnoses_and_ssir.csv',
                       blood_df=None, blood_csv='labevents.csv', get_data_from_df=None, get_data_from='chartevents.csv',
                       output_csv=None, subject_id_column='subject_id', time_column='charttime', itemid_column='itemid',
                       has_sepsis_column='has_sepsis', log_stats=True, sepsis_info_df=None,
                       sepsis_info_csv='sepsis_info_df.csv',
                       valueuom_column='valueuom'):
    """
    Merges diagnoses and SSIR data with blood and chartevents data, and logs statistics about sepsis patients.
    It is recommended to get the file diagnoses_and_ssir by get_diagnoses and merge_diagnoses_and_ssir. 'sepsis_info_df' is recommended to be a path
    to a .csv file after 'get_disease_info' function.

    Args:
        analyzes_names (dict, optional): Dictionary mapping item IDs to analysis names. Default is None, in which case a predefined dictionary is used.
        merge_with_df (pd.DataFrame, optional): DataFrame containing diagnoses and SSIR data. Default is None.
        merge_with (str, optional): Path to the CSV file containing diagnoses and SSIR data. Default is 'diagnoses_and_ssir.csv'.
        blood_df (pd.DataFrame, optional): DataFrame containing blood analysis data. Default is None.
        blood_csv (str, optional): Path to the CSV file containing blood analysis data. Default is 'labevents.csv'.
        get_data_from_df (pd.DataFrame, optional): DataFrame containing chartevents data. Default is None.
        get_data_from (str, optional): Path to the CSV file containing chartevents data. Default is 'chartevents.csv'.
        output_csv (str, optional): Path to the output CSV file for combined data. Default is None.
        subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
        time_column (str): Column name for chart times. Default is 'charttime'.
        itemid_column (str): Column name for item IDs. Default is 'itemid'.
        has_sepsis_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
        log_stats (bool): Whether to log statistics about sepsis patients. Default is True.
        sepsis_info_df (pd.DataFrame, optional): DataFrame containing sepsis information. Default is None.
        sepsis_info_csv (str, optional): Path to the CSV file containing sepsis information. Default is 'sepsis_info_df.csv'.
        valueuom_column (str): Column name for value units of measurement. Default is 'valueuom'.

    Returns:
        pd.DataFrame: The combined and processed data.
    """
    if analyzes_names is None:
        analyzes_names = {
            51222: "Hemoglobin",
            51279: "Red Blood Cell",
            51240: "Large Platelets",
            50861: "Alanine Aminotransferase (ALT)",
            50878: "Aspartate Aminotransferase (AST)",
            225651: "Direct Bilirubin",
            50867: "Amylase",
            51301: "White Blood Cells",
            227444: "C Reactive Protein (CRP)",
            225170: "Platelets",
            220615: "Creatinine (serum)",
            225690: "Total Bilirubin",
            229761: "Creatinine (whole blood)",
            226751: "CreatinineApacheIIScore",
            226752: "CreatinineApacheIIValue",
            227005: "Creatinine_ApacheIV",
            51221: "Hematocrit",
            51265: "Platelet Count",
            51248: "MCH",
            51250: "MCV",
            51249: "MCHC",
            50912: "Creatinine",
            50920: "Estimated GFR (MDRD equation)"
        }

    # Read the data from CSV if DataFrames are not provided
    if merge_with_df is None and merge_with is not None:
        diagnoses_and_ssir = pd.read_csv(merge_with)
    elif merge_with_df is not None:
        diagnoses_and_ssir = merge_with_df
    else:
        raise ValueError("Either merge_with_df or merge_with must be provided.")

    if blood_df is None and blood_csv is not None:
        blood = pd.read_csv(blood_csv)
    elif blood_df is not None:
        blood = blood_df
    else:
        raise ValueError("Either blood_df or blood_csv must be provided.")

    if get_data_from_df is None and get_data_from is not None:
        chartevents = pd.read_csv(get_data_from)
    elif get_data_from_df is not None:
        chartevents = get_data_from_df
    else:
        raise ValueError("Either get_data_from_df or get_data_from must be provided.")

    if sepsis_info_df is None and sepsis_info_csv is not None:
        sepsis_info = pd.read_csv(sepsis_info_csv)
    elif sepsis_info_df is not None:
        sepsis_info = sepsis_info_df
    else:
        raise ValueError("Either sepsis_info_df or sepsis_info_csv must be provided.")

    blood['analysis_name'] = blood[itemid_column].map(analyzes_names)
    pivot_values_blood = blood.pivot_table(index=[subject_id_column, time_column], columns='analysis_name',
                                           values='value', aggfunc='first').reset_index()
    pivot_uom_blood = blood.pivot_table(index=[subject_id_column, time_column], columns='analysis_name',
                                        values=valueuom_column, aggfunc='first').reset_index()

    chartevents['analysis_name'] = chartevents[itemid_column].map(analyzes_names)
    pivot_values_chartevents = chartevents.pivot_table(index=[subject_id_column, time_column], columns='analysis_name',
                                                       values='value', aggfunc='first').reset_index()
    pivot_uom_chartevents = chartevents.pivot_table(index=[subject_id_column, time_column], columns='analysis_name',
                                                    values=valueuom_column, aggfunc='first').reset_index()

    pivot_values = pd.merge(pivot_values_blood, pivot_values_chartevents, on=[subject_id_column, time_column],
                            how='outer')
    pivot_uom = pd.merge(pivot_uom_blood, pivot_uom_chartevents, on=[subject_id_column, time_column], how='outer')

    pivot_uom.columns = [f'{col}_{valueuom_column}' if col not in [subject_id_column, time_column] else col for col in
                         pivot_uom.columns]
    pivot_df = pd.merge(pivot_values, pivot_uom, on=[subject_id_column, time_column], how='left')
    merged_df = pd.merge(pivot_df, diagnoses_and_ssir, on=[subject_id_column, time_column], how='outer')

    sepsis_map = sepsis_info.set_index(subject_id_column)[has_sepsis_column].to_dict()
    merged_df[has_sepsis_column] = merged_df[subject_id_column].map(sepsis_map)
    unique_patients = merged_df[[subject_id_column, has_sepsis_column]].drop_duplicates()
    grouped_sepsis = unique_patients.groupby(subject_id_column)[has_sepsis_column].agg(['min', 'max'])
    ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']].index
    merged_df.loc[merged_df[subject_id_column].isin(ambiguous_sepsis_patients), has_sepsis_column] = False

    if output_csv is not None:
        merged_df.to_csv(output_csv, index=False)

    if log_stats:
        sepsis_counts = unique_patients[has_sepsis_column].value_counts(normalize=False)
        count_with_sepsis = sepsis_counts.get(True, 0)
        count_without_sepsis = sepsis_counts.get(False, 0)
        ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']]
        count_ambiguous_sepsis = len(ambiguous_sepsis_patients)
        print(f'Unique patients with sepsis: {count_with_sepsis}')
        print(f'Unique patients without sepsis: {count_without_sepsis}')
        print(f'Patients with both sepsis and no sepsis records: {count_ambiguous_sepsis}')
        print(f'Total unique patients: {len(grouped_sepsis)}')

    return merged_df

# df_merge_with = pd.read_csv('diagnoses_and_ssir.csv')
# df_blood = pd.read_csv('labevents.csv')
# df_chartevents = pd.read_csv('chartevents.csv')
# df_sepsis_info = pd.read_csv('sepsis_info_df.csv')
# result_df = merge_and_get_data(merge_with_df=df_merge_with, blood_df=df_blood, get_data_from_df=df_chartevents, sepsis_info_df=df_sepsis_info)
# result_df.to_csv('diagnoses_and_ssir_and_blood_and_chartevents.csv', index=False)
