import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
df = pd.read_csv('balance.csv')
df.replace('___', np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
columns_to_fill_with_median = ['Alanine Aminotransferase (ALT)','Amylase','Asparate Aminotransferase (AST)','Hemoglobin','Large Platelets','Red Blood Cell','White Blood Cells','Heart rate','Respiratory rate','Temperature Celsius','Direct Bilurubin']
all_columns = ['Alanine Aminotransferase (ALT)','Amylase','Asparate Aminotransferase (AST)','Hemoglobin','Large Platelets','Red Blood Cell','White Blood Cells','Heart rate','Respiratory rate','Temperature Celsius','Direct Bilurubin', 'has_sepsis']
print("done")
features = df[columns_to_fill_with_median]
scaler = StandardScaler()
df[columns_to_fill_with_median] = scaler.fit_transform(features)
#Образание датасета для тестирования:
# df1 = df.iloc[:10000]
# df2 = df.iloc[-10000:]
# df = pd.concat([df1, df2])
#Выявление уникальных subject_id для корректного разделения на тестовый и тренировочный наборы данных
unique_ids = df['subject_id'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_df = df[df['subject_id'].isin(train_ids)]
test_df = df[df['subject_id'].isin(test_ids)]
X_train = train_df[columns_to_fill_with_median]
y_train = train_df['has_sepsis']
X_test = test_df[columns_to_fill_with_median]
y_test = test_df['has_sepsis']
df = df[all_columns]
#Подбор гиперпараметров через GridSearch:
# dt_params = {'max_depth': [3, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
# rf_params = {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
# gb_params = {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 10, 15]}
# xgb_params = {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 10, 15]}
# cb_params = {'iterations': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.5], 'depth': [3, 5, 8]}

# dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
# dt_grid.fit(X_train, y_train)
# print("Лучшие параметры для DecisionTreeClassifier:", dt_grid.best_params_)
# rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
# rf_grid.fit(X_train, y_train)
# print("Лучшие параметры для RandomForestClassifier:", rf_grid.best_params_)
# gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_params, cv=5)
# gb_grid.fit(X_train, y_train)
# print("Лучшие параметры для GradientBoostingClassifier:", gb_grid.best_params_)
# xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), xgb_params, cv=5)
# xgb_grid.fit(X_train, y_train)
# print("Лучшие параметры для XGBClassifier:", xgb_grid.best_params_)
# cb_grid = GridSearchCV(CatBoostClassifier(verbose=False), cb_params, cv=5)
# cb_grid.fit(X_train, y_train)
# print("Лучшие параметры для CatBoostClassifier:", cb_grid.best_params_)

#Обучение моделей
print("Learning")
print(X_train)
#model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=10)
model = CatBoostClassifier(n_estimators=200, learning_rate=0.25, max_depth=10)
#model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Predicting")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred, normalize=False)
print(f"Точность модели: {accuracy}")
print(f"Всего правильных предсказаний: {acc} из {len(y_test)}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)



