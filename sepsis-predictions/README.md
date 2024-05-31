# Predictions sepsis

## Instruction

Predictions sepsis is a module based on pandas, torch, and scikit-learn that allows users to perform simple operations with the MIMIC dataset.
With this module, using just a few functions, you can train your model to predict whether some patients have certain diseases or not. 
By default, the module is designed to train and predict sepsis. 
The module also allows users to change different names of tables to aggregate data from.

### Installation

To install the module, use the following command:

```bash
pip install predictions-sepsis
```
or
```bash
pip3 install predictions-sepsis
```
### Usage

You can import functions from the module into your Python file to aggregate data from MIMIC, 
fill empty spots, compress data between patients, and train your model.

### Examples

#### Aggregate patient diagnoses Data
```python
import predictions_sepsis as ps

ps.get_diagnoses(patient_diagnoses_csv='path_to_patient_diagnoses.csv', 
                 all_diagnoses_csv='path_to_all_diagnoses.csv',
                 output_file_csv='gottenDiagnoses.csv')
```

#### Aggregate patient ssir Data
```python
import predictions_sepsis as ps

ps.get_ssir(chartevents_csv='chartevents.csv', subject_id_col='subject_id', itemid_col='itemid',
             charttime_col='charttime', value_col='value', valuenum_col='valuenum', valueuom_col='valueuom',
             itemids=None, rest_columns=None, output_csv='ssir.csv'):
```

#### Combine Diagnoses and SSIR Data
```python
import predictions_sepsis as ps

ps.combine_diagnoses_and_ssir(gotten_diagnoses_csv='gottenDiagnoses.csv', 
                              ssir_csv='path_to_ssir.csv',
                              output_file='diagnoses_and_ssir.csv')
```

#### Aggregate patient blood analysis data from chartevents.csv and labevents.csv and combine it with diagnoses and SSIR Data
```python
import predictions_sepsis as ps

ps.merge_diagnoses_and_ssir_with_blood(diagnoses_and_ssir_csv='diagnoses_and_ssir.csv', 
                                       blood_csv='path_to_blood.csv',
                                       chartevents_csv='path_to_chartevents.csv',
                                       output_csv='merged_data.csv')
)
```

#### Compress Data by patient
```python
import predictions_sepsis as ps

ps.compress(df_to_compress='balanced_data.csv', 
            output_csv='compressed_data.csv')

```

#### Choose top non-sepsis patients to balance
```python
import predictions_sepsis as ps

ps.choose(compressed_df_csv='compressed_data.csv', 
          output_file='final_balanced_data.csv')
```

#### Fill missing values with mode
```python
import predictions_sepsis as ps

ps.fill_values(balanced_csv='final_balanced_data.csv', 
               strategy='most_frequent', 
               output_csv='filled_data.csv')
```

#### Aggregate patient diagnoses Data
```python
import predictions_sepsis as ps

# Aggregate diagnoses data
ps.get_diagnoses(patient_diagnoses_csv='path_to_patient_diagnoses.csv', 
                 all_diagnoses_csv='path_to_all_diagnoses.csv',
                 output_file_csv='gottenDiagnoses.csv')
```

#### Train model
```python
import predictions_sepsis as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
model = ps.train_model(df_to_train_csv='filled_data.csv', 
                       categorical_col=['Large Platelets'], 
                       columns_to_train_on=['Amylase'], 
                       model=RandomForestClassifier(), 
                       single_cat_column='White Blood Cells', 
                       has_disease_col='has_sepsis', 
                       subject_id_col='subject_id', 
                       valueuom_col='valueuom', 
                       scaler=MinMaxScaler(), 
                       random_state=42, 
                       test_size=0.2)
```




