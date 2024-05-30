import pandas as pd


def combine_diagnoses_and_ssir(gotten_diagnoses_csv='gottenDiagnoses.csv', has_sepsis_column='has_sepsis',
                               ssir_csv='ssir.csv', title_column='long_title'):
    def fahrenheit_to_celsius(f):
        return (f - 32) * 5.0 / 9.0

    diagnoses = pd.read_csv(gotten_diagnoses_csv)
    ssir = pd.read_csv(ssir_csv)

    diagnoses[has_sepsis_column] = diagnoses['long_title'].str.contains('sepsis', case=False, na=False)
    sepsis_info_df = diagnoses.groupby('subject_id')[has_sepsis_column].any().reset_index()
    sepsis_info_df.to_csv('sepsis_info_df.csv')
    ans = sepsis_info_df[has_sepsis_column].sum()
    print(f'Количество пациентов с сепсисом правильное: {ans}')
    merged_df = pd.merge(ssir, sepsis_info_df, on='subject_id', how='left')
    merged_df.drop(columns=[col for col in merged_df.columns if 'valueom' in col], inplace=True)
    merged_df['Temperature Celsius'] = merged_df.apply(
        lambda row: fahrenheit_to_celsius(row['Temperature Fahrenheit']) if pd.notnull(
            row['Temperature Fahrenheit']) else
        row['Temperature Celsius'],
        axis=1
    )
    merged_df.drop(columns=['Temperature Fahrenheit'], inplace=True)
    merged_df.to_csv('diagnoses_and_ssir.csv', index=False)

    unique_patients = merged_df[['subject_id', has_sepsis_column]].drop_duplicates()
    sepsis_counts = unique_patients[has_sepsis_column].value_counts(normalize=False)
    count_with_sepsis = sepsis_counts.get(True, 0)
    count_without_sepsis = sepsis_counts.get(False, 0)

    grouped_sepsis = unique_patients.groupby('subject_id')[has_sepsis_column].agg(['min', 'max'])
    ambiguous_sepsis_patients = grouped_sepsis[grouped_sepsis['min'] != grouped_sepsis['max']]
    count_ambiguous_sepsis = len(ambiguous_sepsis_patients)

    print(f'Unique patients with predictions_sepsis: {count_with_sepsis}')
    print(f'Unique patients without predictions_sepsis: {count_without_sepsis}')
    print(f'Patients with both predictions_sepsis and no predictions_sepsis records: {count_ambiguous_sepsis}')
    print(f'Всего уникальных пациентов: {len(grouped_sepsis)}')
