# Disease-predictions module

## Instruction

Predictions sepsis is a module based on pandas, torch, and scikit-learn that allows users to perform simple operations with the MIMIC dataset.
With this module, using just a few functions, you can train your model to predict whether some patients have certain diseases or not. 
By default, the module is designed to train and predict sepsis. 
The module also allows users to change different names of tables to aggregate data from.

### Installation

To install the module, use the following command:

```bash
pip install sickness-screening
```
or
```bash
pip3 install sickness-screening
```
### Usage

You can import functions from the module into your Python file to aggregate data from MIMIC, 
fill empty spots, compress data between patients, and train your model.

### Examples
#### MIMIC Setup
In the examples, we will show how to use the sickness-screening module to train a model to predict sepsis on the MIMIC dataset.
MIMIC contains many tables, but for the example, we will need the following tables:
1. **chartevents.csv** -- contains patient monitoring data, such as body temperature and blood pressure.
2. **labevents.csv** -- contains various patient test data, such as different blood test characteristics for patients.
3. **diagnoses.csv** -- contains information about the diagnoses received by the patient.
4. **d_icd_diagnoses.csv** -- decoding of diagnosis codes for each diagnosis.
5. **d_labitems.csv** -- decoding of test codes for each patient.
#### Aggregating patient diagnosis data:
First, we will collect data on patient diagnoses:
```python
import sickness_screening as ss

ss.get_diagnoses_data(patient_diagnoses_csv='diagnoses.csv', 
                 all_diagnoses_csv='d_icd_diagnoses.csv',
                 output_file_csv='gottenDiagnoses.csv')
```
Here, for each patient from **patient_diagnoses_csv**, we get the diagnosis codes, and then, using **all_diagnoses_csv**,
we get the output_file_csv file, which stores the decoding of each patient's diagnosis.
#### Obtaining data on whether a specific diagnosis is present in a patient
```python
import sickness_screening as ss
ss.get_diseas_info(diagnoses_csv='gottenDiagnoses.csv', title_column='long_title', diseas_str='sepsis',
                    diseas_column='has_sepsis', subject_id_column='subject_id', log_stats=True,
                    output_csv='sepsis_info.csv')
```
Here we use the table obtained from the previous example to get a table containing data on whether the patient's diagnosis contains the substring sepsis or not.
#### Aggregating data needed to determine SIRS (systemic inflammatory response syndrome)
Now we will collect some data needed to determine SIRS:
```python
import sickness_screening as ss

ss.get_analyzes_data(analyzes_csv='chartevents.csv', subject_id_col='subject_id', itemid_col='itemid',
                  charttime_col='charttime', value_col='value', valuenum_col='valuenum',
                  itemids=[220045, 220210, 223762, 223761, 225651], rest_columns=['Heart rate', 'Respiratory rate', 'Temperature Fahrenheit', 'Temperature Celsius',
                    'Direct Bilirubin'], output_csv='ssir.csv')

```
Here we use the **analyzes_csv** table, **itemids** (the codes of the tests we want to collect), and **rest_columns** (the columns we want to keep in the output table).
The function collects measurements for patients with **itemids** codes from analyzes_csv and writes them to **output_csv**, keeping only the columns present in **rest_columns**.
In this function, **subject_id_col** and **itemid_col** are responsible for the columns assigned to patient and test codes, respectively.
**charttime_col** is responsible for the time. **valuenum_col** is responsible for the column with test measurement units.
#### Combining diagnosis and SIRS data
Now we will combine the data from the previous two examples into one table:
```python
import sickness_screening as ss

ss.combine_data(first_data='gottenDiagnoses.csv', 
                              second_data='ssir.csv',
                              output_file='diagnoses_and_ssir.csv')
```
#### Collecting and combining blood test data with diagnosis and SIRS data
We will collect patient blood test data and combine them into one table:
```python
import sickness_screening as ss

ss.merge_and_get_data(merge_with='diagnoses_and_ssir.csv', 
                      blood_csv='labevents.csv',
                      get_data_from='chartevents.csv',
                      output_csv='merged_data.csv',
                      analyzes_names = {
                        51222: "Hemoglobin",
                        51279: "Red Blood Cell",
                        51240: "Large Platelets",
                        50861: "Alanine Aminotransferase (ALT)",
                        50878: "Aspartate Aminotransferase (AST)",
                        225651: "Direct Bilirubin",
                        50867: "Amylase",
                        51301: "White Blood Cells"})
```
This function searches for data on analyzes_names for patients from the blood_csv and **get_data_from** tables,
combines them with **merge_with**. Note that this function also combines disease data for each patient.
#### Balancing data within each patient
We will balance the data by the total number of rows for patients with and without sepsis.
```python 
import sickness_screening as ss
ss.balance_on_patients(balancing_csv='merged_data.csv', disease_col='has_sepsis', subject_id_col='subject_id',
                        output_csv='balance.csv',
                        output_filtered_csv='balance_filtered.csv',
                        filtering_on=200,
                        number_of_patient_selected=50000,
                        log_stats=True)
```
#### Compressing data for each patient (if there are gaps in the dataset, the gaps within each patient will be filled with the patient's own values)
Now we will fill the gaps with the available data for each patient without filling with statistical values or constants:
```python
import sickness_screening as ss

ss.compress(df_to_compress='balanced_data.csv', 
            subject_id_col='subject_id',
            output_csv='compressed_data.csv')
```
#### Select the best patients with data for final balancing
```python
import sickness_screening as ss

ss.choose(compressed_df_csv='compressed_data.csv', 
          output_file='final_balanced_data.csv')
```
#### Filling missing values with the most frequent value
```python
import sickness_screening as ss

ss.fill_values(balanced_csv='final_balanced_data.csv', 
               strategy='most_frequent', 
               output_csv='filled_data.csv')
```
#### Training the model on the dataset
```python
import sickness_screening as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
model = ss.train_model(df_to_train_csv='filled_data.csv', 
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
In this function, we train a RandomForestClassifier from scikit-learn on a dataset with one categorical column, one numeric column,
and one categorical column that can be converted to numeric. MinMaxScaler from scikit-learn is used as the normalization method.
#### For example, you can insert models like CatBoostClassifier or SVC with different kernels.
CatBoostClassifier:
```python
class_weights = {0: 1, 1: 15}
clf = CatBoostClassifier(loss_function='MultiClassOneVsAll', class_weights=class_weights, iterations=50, learning_rate=0.1, depth=5)
clf.fit(X_train, y_train)
```
SVC using Gaussian kernel with radial basis function (RBF):
```python
class_weights = {0: 1, 1: 13}
param_dist = {
    'C': reciprocal(0.1, 100),
    'gamma': reciprocal(0.01, 10),
    'kernel': ['rbf']
}

svm_model = SVC(class_weight=class_weights, random_state=42)
random_search = RandomizedSearchCV(
    svm_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring=make_scorer(recall_score, pos_label=1),
    n_jobs=-1
)
```

## The Second Method (Transformers TabNet and DeepFM)
### Collecting features into a dataset
#### You can choose any features, but we will take 4 as in MEWS (Modified Early Warning Score) to predict sepsis in the first hours of a patient's hospital stay:
* Systolic blood pressure
* Heart rate
* Respiratory rate
* Temperature
```python
  item_ids_set = set(item_ids)

  with open(file_path) as f:
      headers = f.readline().replace('\n', '').split(',')
      i = 0
      for line in tqdm(f):
          values = line.replace('\n', '').split(',')
          subject_id = values[0]
          item_id = values[6]
          valuenum = values[8]
          if item_id in item_ids_set:
              if subject_id not in result:
                  result[subject_id] = {}
              result[subject_id][item_id] = valuenum
          i += 1
  
  table = pd.DataFrame.from_dict(result, orient='index')
  table['subject_id'] = table.index

item_ids = [str(x) for x in [225309, 220045, 220210, 223762]]
```

#### Adding the target
```python
target_subjects = drgcodes.loc[drgcodes['drg_code'].isin([870, 871, 872]), 'subject_id']
merged_data.loc[merged_data['subject_id'].isin(target_subjects), 'diagnosis'] = 1
```

#### Filling in gaps using the NoNa library. This algorithm fills in gaps using various machine learning methods, we use StandardScaler, Ridge and RandomForestClassifier
```python
nona(
    data=X,
    algreg=make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1)),
    algclass=RandomForestClassifier(max_depth=2, random_state=0)
)
```

#### Addressing class imbalance using SMOTE
```python
smote = SMOTE(random_state=random_state)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Training the TabNet model. TabNet is an extension of pyTorch. First, we use semi-supervised pretraining with TabNetPretrainer, then create and train a classification model using TabNetClassifier
```python
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=pretraining_lr),
    mask_type=mask_type
)

unsupervised_model.fit(
    X_train=X_train.values,
    eval_set=[X_val.values],
    pretraining_ratio=pretraining_ratio,
)

clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=training_lr),
    scheduler_params=scheduler_params,
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type=mask_type
)

clf.fit(
    X_train=X_train.values, y_train=y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_metric=['auc'],
    max_epochs=max_epochs,
    patience=patience,
    from_unsupervised=unsupervised_model
)
```

#### Training the DeepFM model
```python
deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                lr=1e-4, lr_decay=False, reg=None, batch_size=1,
                num_neg=1, use_bn=False, dropout_rate=None,
                hidden_units="128,64,32", tf_sess_config=None)

deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
           metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                    "precision", "recall", "map", "ndcg"])
```

#### Viewing the obtained metrics
```python
result = loaded_clf.predict(X_test.values)
accuracy = (result == y_test.values).mean()
precision = precision_score(y_test.values, result)
recall = recall_score(y_test.values, result)
f1 = f1_score(y_test.values, result)
```

#### Visualization of 2 PCA components was performed
![Image alt](./Визуализация_2_PCA_компоненты.png)
The distribution by components is presented below:

|                  |  Load on the first component  | Load on the second component  |
| ---------------- | :---: | :---: |
| Heart rate       |           -0.101450           |            0.991611           |
| Temperature      |            0.001178           |            0.013098           |
| Systolic BP      |            0.994771           |            0.100169           |
| Respiratory rate |            0.011673           |            0.080573           |
| MEWS             |           -0.000660           |            0.003313           |

No patterns were found.

#### A variational encoder was trained to build a separable 2D space
![Image alt](./Вариационный_кодировщик.png)
We can see that they overlap and are inseparable.
