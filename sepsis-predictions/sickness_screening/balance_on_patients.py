import pandas as pd
import numpy as np


def balance_on_patients(balancing_csv='final.csv', disease_col='has_sepsis', subject_id_col='subject_id',
                        output_csv='balance.csv',
                        output_filtered_csv='balance_filtered.csv',
                        filtering_on=200,
                        number_of_patient_selected=50000,
                        log_stats=True
                        ):
    """
        Balances the dataset by selecting a specified number of non-sepsis patients and combines them with disease patients.
        Filters the resulting dataset to include only patients with at least 'filtering_on' records.
        The dataset in disease column should have 0 or 1 value: either patient has disease or not.

        Args:
            balancing_csv (str): Path to the CSV file containing the data to balance. Default is 'final.csv'.
            disease_col (str): Column name indicating the presence of the disease (1 for sepsis, 0 for non-sepsis). Default is 'has_sepsis'.
            subject_id_col (str): Column name for subject IDs. Default is 'subject_id'.
            output_csv (str): Path to the output CSV file for the balanced dataset. Default is 'balance.csv'.
            output_filtered_csv (str): Path to the output CSV file for the filtered balanced dataset. Default is 'balance_filtered.csv'.
            log_stats (bool): Whether to log statistics about the dataset. Default is True.
            filtering_on (int): Number of lines that patient is required to stay in filtered dataset.
            number_of_patient_selected (int): Number of patient that are being selected from non_disease group in order to complete balancing.
        Returns:
            None: Writes the balanced and filtered datasets to the specified CSV files.
        """
    df = pd.read_csv(balancing_csv)

    sepsis_patients = df[df[disease_col] == 1]
    non_sepsis_patients = df[df[disease_col] == 0]

    unique_non_sepsis_subject_ids = non_sepsis_patients[subject_id_col].unique()
    selected_subject_ids = np.random.choice(unique_non_sepsis_subject_ids, number_of_patient_selected, replace=False)

    selected_patients_data = df[df[subject_id_col].isin(selected_subject_ids)]

    final_dataset = pd.concat([sepsis_patients, selected_patients_data])
    counts = final_dataset[subject_id_col].value_counts()
    filtered_dataset = final_dataset[final_dataset[subject_id_col].isin(counts[counts >= filtering_on].index)]

    num_sepsis = filtered_dataset[filtered_dataset[disease_col] == 1][subject_id_col].nunique()
    num_non_sepsis = filtered_dataset[filtered_dataset[disease_col] == 0][subject_id_col].nunique()
    num_rows_sepsis = filtered_dataset[filtered_dataset[disease_col] == 1].shape[0]
    num_rows_non_sepsis = filtered_dataset[filtered_dataset[disease_col] == 0].shape[0]
    num_rows_sepsis_final = final_dataset[final_dataset[disease_col] == 1].shape[0]
    num_rows_non_sepsis_final = final_dataset[final_dataset[disease_col] == 0].shape[0]
    if log_stats:
        print(f"In filtered dataset:")
        print(f"Unique patients with disease: {num_sepsis}")
        print(f"Unique patients without disease {num_non_sepsis}")
        print(f"Number of lines for patients with disease {num_rows_sepsis}")
        print(f"Number of lines for patients without disease {num_rows_non_sepsis}")
        print(f"\nIn original dataset")
        print(f"Number of line for patients with disease {num_rows_sepsis_final}")
        print(f"Number of lines for patients without disease {num_rows_non_sepsis_final}")

    filtered_dataset.to_csv(output_filtered_csv, index=False)
    final_dataset.to_csv(output_csv, index=False)

