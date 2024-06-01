# Модуль Disease-predictions

## Инструкция

Модуль позволяет пользователям взаимодействовать с различными медицинскими таблицами.
Он упрощает агрегацию, балансировку, заполнение пустых значений данных, связанных с конкретным пациентом.
Также позволяет достаточно быстро и просто получить статистики по данным, нормализовать их и обучить на них модель.

### Установка модуля

Чтобы установить модуль, достаточно воспользоваться следующей командой:

```bash
pip install predictions-sepsis
```
или
```bash
pip3 install predictions-sepsis
```
### Использование
После импортирования модуля в своем файле появляется возможность взаимодействовать с различными медицинскими данными. 
По умолчанию модуль настроен на набор данных MIMIC и предсказание сепсиса. 

### Примеры

#### Аггрегирование данных о диагнозах пациентов:
```python
import predictions_sepsis as ps

ps.get_diagnoses(patient_diagnoses_csv='path_to_patient_diagnoses.csv', 
                 all_diagnoses_csv='path_to_all_diagnoses.csv',
                 output_file_csv='gottenDiagnoses.csv')
```
#### Аггрегирование данных, необходимых для нахождения ССВР (синдром системной воспалительной рекции)
```python
import predictions_sepsis as ps

ps.get_analasys_data(chartevents_csv='chartevents.csv', subject_id_col='subject_id', itemid_col='itemid',
             charttime_col='charttime', value_col='value', valuenum_col='valuenum', valueuom_col='valueuom',
             itemids=None, rest_columns=None, output_csv='ssir.csv')
```

#### Комбинирование данных о диагнозах и ССВР
```python
import predictions_sepsis as ps

ps.combine_diagnoses_and_ssir(gotten_diagnoses_csv='gottenDiagnoses.csv', 
                              ssir_csv='path_to_ssir.csv',
                              output_file='diagnoses_and_ssir.csv')
```

#### Сбор и комбинирование данных об анализах крови, с данными об диагнозах и ССВР
```python
import predictions_sepsis as ps

ps.merge_diagnoses_and_ssir_with_blood(diagnoses_and_ssir_csv='diagnoses_and_ssir.csv', 
                                       blood_csv='path_to_blood.csv',
                                       chartevents_csv='path_to_chartevents.csv',
                                       output_csv='merged_data.csv')
```

#### Компрессия данных о каждом пациенте (если в наборе данных пропуски, то внутри каждого пациента пропуски заполнятся значением из этого пациента)
```python
import predictions_sepsis as ps

ps.compress(df_to_compress='diagnoses_and_ssir_and_blood_and_chartevents.csv', subject_id_col='subject_id',
             output_csv='compressed.csv')

```

#### Выбрать лучших пациентов с данными для балансировки
```python
import predictions_sepsis as ps

ps.choose(compressed_df_csv='compressed_data.csv', 
          output_file='final_balanced_data.csv')
```

#### Заполнение пропущенных значений модой
```python
import predictions_sepsis as ps

ps.fill_values(balanced_csv='final_balanced_data.csv', 
               strategy='most_frequent', 
               output_csv='filled_data.csv')
```

#### Тренировка модели на наборе данных.
```python
import predictions_sepsis as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
model = ps.train_model(df_to_train_csv='filled_data.csv', 
                       categorical_col=['Large Platelets'], #Категориальная колонка
                       columns_to_train_on=['Amylase'], #Числовая колонка
                       model=RandomForestClassifier(), 
                       single_cat_column='White Blood Cells', 
                       has_disease_col='has_sepsis', 
                       subject_id_col='subject_id', 
                       valueuom_col='valueuom', 
                       scaler=MinMaxScaler(), 
                       random_state=42, 
                       test_size=0.2)
```

## Second way
#### Collecting features of the dataset
```python
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

table.to_csv(output_path, index=False)
```

#### Add a target to the dataset
```python
target_subjects = drgcodes.loc[drgcodes['drg_code'].isin([870, 871, 872]), 'subject_id']
merged_data.loc[merged_data['subject_id'].isin(target_subjects), 'diagnosis'] = 1
```

#### Filling in the blanks using the NoNa library
```python
nona(
    data=X,
    algreg=make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1)),
    algclass=RandomForestClassifier(max_depth=2, random_state=0)
)
```

#### Removing class imbalance using SMOTE
```python
smote = SMOTE(random_state=random_state)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Train model TabNet
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

#### Looking at the metrics
```python
result = loaded_clf.predict(X_test.values)
accuracy = (result == y_test.values).mean()
precision = precision_score(y_test.values, result)
recall = recall_score(y_test.values, result)
f1 = f1_score(y_test.values, result)
```
