import pandas as pd


def merge_diagnoses_and_ssir_with_blood(analysis_names=None, diagnoses_and_ssir_csv='diagnoses_and_ssir.csv',
                                        blood_csv='labevents.csv', chartevents_csv='chartevents.csv',
                                        output_csv='diagnoses_and_ssir_and_blood_and_chartevents.csv',
                                        subject_id_column='subject_id',
                                        charttime_column='charttime',
                                        itemid_column='itemid',
                                        has_sepsis_column='has_sepsis',
                                        log_stats=True,
                                        sepsis_info_df='sepsis_info_df.csv',
                                        valueuom_column='valueuom'
                                        ):
    """
        Merges diagnoses and SSIR data with blood and chartevents data, and logs statistics about sepsis patients.
        It is recommended to get file diagnoses_and_ssir by get_diagnoses and merge_diagnoses_and_ssir. 'sepsis_info_df' is recommended to be a path
        to a .csv file after 'get_disease_info' function.
        Args:
            analysis_names (dict, optional): Dictionary mapping item IDs to analysis names. Default is None, in which case a predefined dictionary is used.
            diagnoses_and_ssir_csv (str): Path to the CSV file containing diagnoses and SSIR data. Default is 'diagnoses_and_ssir.csv'.
            blood_csv (str): Path to the CSV file containing blood analysis data. Default is 'labevents.csv'.
            chartevents_csv (str): Path to the CSV file containing chartevents data. Default is 'chartevents.csv'.
            output_csv (str): Path to the output CSV file for combined data. Default is 'diagnoses_and_ssir_and_blood_and_chartevents.csv'.
            subject_id_column (str): Column name for subject IDs. Default is 'subject_id'.
            charttime_column (str): Column name for chart times. Default is 'charttime'.
            itemid_column (str): Column name for item IDs. Default is 'itemid'.
            has_sepsis_column (str): Column name to indicate sepsis presence. Default is 'has_sepsis'.
            log_stats (bool): Whether to log statistics about sepsis patients. Default is True.
            sepsis_info_df (str): Path to the CSV file containing sepsis information. Default is 'sepsis_info_df.csv'.
            valueuom_column (str): Column name for value units of measurement. Default is 'valueom'.
        Returns:
            None: Writes the combined and processed data to the specified output file and logs statistics if required.
        """
    if analysis_names is None:
        analysis_names = {
            51222: "Hemoglobin",
            51279: "Red Blood Cell",
            51240: "Large Platelets",
            50861: "Alanine Aminotransferase (ALT)",
            50878: "Asparate Aminotransferase (AST)",
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
    diagnoses_and_ssir = pd.read_csv(diagnoses_and_ssir_csv)
    blood = pd.read_csv(blood_csv)
    chartevents = pd.read_csv(chartevents_csv)

    blood['analysis_name'] = blood[itemid_column].map(analysis_names)

    pivot_values_blood = blood.pivot_table(index=[subject_id_column, charttime_column], columns='analysis_name',
                                           values='value',
                                           aggfunc='first').reset_index()
    pivot_uom_blood = blood.pivot_table(index=[subject_id_column, charttime_column], columns='analysis_name',
                                        values= valueuom_column,
                                        aggfunc='first').reset_index()
    chartevents_itemids = analysis_names
    chartevents['analysis_name'] = chartevents[itemid_column].map(chartevents_itemids)

    pivot_values_chartevents = chartevents.pivot_table(index=[subject_id_column, charttime_column],
                                                       columns='analysis_name',
                                                       values='value', aggfunc='first').reset_index()
    pivot_uom_chartevents = chartevents.pivot_table(index=[subject_id_column, charttime_column],
                                                    columns='analysis_name',
                                                    values=valueuom_column, aggfunc='first').reset_index()

    pivot_values = pd.merge(pivot_values_blood, pivot_values_chartevents, on=[subject_id_column, charttime_column],
                            how='outer')
    pivot_uom = pd.merge(pivot_uom_blood, pivot_uom_chartevents, on=[subject_id_column, charttime_column], how='outer')

    pivot_uom.columns = [f'{col}_{valueuom_column}' if col not in [subject_id_column, charttime_column] else col for col
                         in
                         pivot_uom.columns]
    pivot_df = pd.merge(pivot_values, pivot_uom, on=[subject_id_column, charttime_column], how='left')
    merged_df = pd.merge(pivot_df, diagnoses_and_ssir, on=[subject_id_column, charttime_column], how='outer')
    sepsis_info_df = pd.read_csv(sepsis_info_df)
    sepsis_map = sepsis_info_df.set_index(subject_id_column)[has_sepsis_column].to_dict()
    merged_df[has_sepsis_column] = merged_df[subject_id_column].map(sepsis_map)
    unique_patients = merged_df[[subject_id_column, has_sepsis_column]].drop_duplicates()
    grouped_sepsis = unique_patients.groupby(subject_id_column)[has_sepsis_column].agg(['min', 'max'])
    ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']].index
    merged_df.loc[merged_df[subject_id_column].isin(ambiguous_sepsis_patients), has_sepsis_column] = False
    merged_df.to_csv(output_csv, index=False)
    unique_patients = merged_df[[subject_id_column, has_sepsis_column]].drop_duplicates()
    sepsis_counts = unique_patients[has_sepsis_column].value_counts(normalize=False)
    count_with_sepsis = sepsis_counts.get(True, 0)
    count_without_sepsis = sepsis_counts.get(False, 0)
    grouped_sepsis = unique_patients.groupby(subject_id_column)[has_sepsis_column].agg(['min', 'max'])
    ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']]
    count_ambiguous_sepsis = len(ambiguous_sepsis_patients)
    if (log_stats):
        print(f'Unique patients with sepsis: {count_with_sepsis}')
        print(f'Unique patients without sepsis: {count_without_sepsis}')
        print(f'Patients with both sepsis and no sepsis records: {count_ambiguous_sepsis}')
        print(f'Total unique patients: {len(grouped_sepsis)}')


