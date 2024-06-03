# Модуль Disease-predictions  
[English](sepsis-predictions/README.md)

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
import sickness_screening as ss

ss.get_diagnoses_data(patient_diagnoses_csv='path_to_patient_diagnoses.csv', 
                 all_diagnoses_csv='path_to_all_diagnoses.csv',
                 output_file_csv='gottenDiagnoses.csv')
```
#### Аггрегирование данных, необходимых для нахождения ССВР (синдром системной воспалительной рекции)
```python
import sickness_screening as ss

ss.get_analyzes_data(analyzes_csv='chartevents.csv', subject_id_col='subject_id', itemid_col='itemid',
                      charttime_col='charttime', value_col='value', valuenum_col='valuenum', valueuom_col='valueuom',
                      itemids=None, rest_columns=None, output_csv='ssir.csv')
```

#### Комбинирование данных о диагнозах и ССВР
```python
import sickness_screening as ss

ss.combine_data(first_data='gottenDiagnoses.csv', 
                              second_data='ssir.csv',
                              output_file='diagnoses_and_ssir.csv')
```

#### Сбор и комбинирование данных об анализах крови, с данными об диагнозах и ССВР
```python
import sickness_screening as ss

ss.merge_and_get_data(merge_with='diagnoses_and_ssir.csv', 
                                       blood_csv='path_to_blood.csv',
                                       get_data_from='path_to_chartevents.csv',
                                       output_csv='merged_data.csv')
```

#### Компрессия данных о каждом пациенте (если в наборе данных пропуски, то внутри каждого пациента пропуски заполнятся значением из этого пациента)
```python
import sickness_screening as ss

ss.compress(df_to_compress='balanced_data.csv', 
            output_csv='compressed_data.csv')

```

#### Выбрать лучших пациентов с данными для балансировки
```python
import sickness_screening as ss

ss.choose(compressed_df_csv='compressed_data.csv', 
          output_file='final_balanced_data.csv')
```

#### Заполнение пропущенных значений модой
```python
import sickness_screening as ss

ss.fill_values(balanced_csv='final_balanced_data.csv', 
               strategy='most_frequent', 
               output_csv='filled_data.csv')
```

#### Тренировка модели на наборе данных
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

#### Например, можно вставить такие модели, как CatBoostClassifier или SVC с разными ядрами
CatBoostClassifier:
```python
class_weights = {0: 1, 1: 15}
clf = CatBoostClassifier(loss_function='MultiClassOneVsAll', class_weights=class_weights, iterations=50, learning_rate=0.1, depth=5)
clf.fit(X_train, y_train)
```
SVC с использованием гауссова ядра с радиальной базовой функцией (RBF):
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

## Второй способ (трансформеры TabNet и DeepFM)
### Собираем признаки в датасет 
#### Можно выбрать абсолютно любые признаки, но мы возьмем 4 как в MEWS (Модифицированная оценка раннего предупреждения), чтобы предсказывать сепсис в первые часы пребывания человека в больнице:
* Систолическое артериальное давление
* Частота сердцебиения
* Частота дыхания
* Температура
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

#### Добавляем таргет
```python
target_subjects = drgcodes.loc[drgcodes['drg_code'].isin([870, 871, 872]), 'subject_id']
merged_data.loc[merged_data['subject_id'].isin(target_subjects), 'diagnosis'] = 1
```

#### Заполнение пробелов с помощью библиотеки NoNa. Данный алгоритм заполняет пропуски различными методами машинного обучения, мы используем StandardScaler, Ridge и RandomForestClassifier
```python
nona(
    data=X,
    algreg=make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1)),
    algclass=RandomForestClassifier(max_depth=2, random_state=0)
)
```

#### Устранение дисбаланса классов с помощью SMOTE
```python
smote = SMOTE(random_state=random_state)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Обучаем модель TabNet. TabNet - это расширение pyTorch. Сначала импользуем полуконтролируемое предварительное обучение с помощью TabNetPretrainer, а далее создаём и обучаем модель классификации с использованием TabNetClassifier
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
#### Обучаем модель DeepFM
```python
deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=2,
                lr=1e-4, lr_decay=False, reg=None, batch_size=1,
                num_neg=1, use_bn=False, dropout_rate=None,
                hidden_units="128,64,32", tf_sess_config=None)

deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
           metrics=["loss", "balanced_accuracy", "roc_auc", "pr_auc",
                    "precision", "recall", "map", "ndcg"])
```

#### Смотрим полученные метрики
```python
result = loaded_clf.predict(X_test.values)
accuracy = (result == y_test.values).mean()
precision = precision_score(y_test.values, result)
recall = recall_score(y_test.values, result)
f1 = f1_score(y_test.values, result)
```

#### Была произведена визуализация по 2 PCA компонентам
![Image alt](./Визуализация_2_PCA_компоненты.png)
Распределение по компонентам представлено ниже:

|                  | Нагрузка на первую компоненту | Нагрузка на вторую компоненту |
| ---------------- | :---: | :---: |
| Heart rate       |           -0.101450           |            0.991611           |
| Temperature      |            0.001178           |            0.013098           |
| Systolic BP      |            0.994771           |            0.100169           |
| Respiratory rate |            0.011673           |            0.080573           |
| MEWS             |           -0.000660           |            0.003313           |

Найти закономерностей не получилось.

#### Обучен вариационный кодировщик для построения разделимого 2D пространства.
![Image alt](./Вариационный_кодировщик.png)
Можем заметить, что они накладываются друг на друга и неразделимы.
