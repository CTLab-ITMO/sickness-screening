import pytest
import pandas as pd
from pathlib import Path
from sickness_screening import (
    process_features, add_diagnosis_column, impute_data, 
    prepare_and_save_data, resample_test_val_data, train_tabnet_model, 
    evaluate_tabnet_model
)

@pytest.fixture
def mock_data():
    chartevents = pd.DataFrame({
        'subject_id': [1, 16, 17, 10, 11, 2, 12, 3, 13, 14, 4, 5, 15, 6, 7, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'hadm_id': [101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101, 101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101],
        'stay_id': [1001, 1001, 1002, 1002, 1001, 1001, 1002, 1002, 1002, 1001, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 101, 102, 101, 101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101],
        'caregiver_id': [1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 101, 102, 101, 101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101],
        'charttime': ['2023-06-01 00:00:00', '2023-06-01 00:10:00', '2023-06-01 00:20:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', '2023-06-01 00:30:00', 101, 102, 101, 101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101],
        'storetime': ['2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', '2180-07-23 22:15:00', 101, 102, 101, 101, 101, 102, 102, 101, 101, 102, 102, 101, 101, 102, 102, 101, 102, 101, 102, 101, 101, 102, 101],
        'item_id': [225309, 220045, 220210, 223762, 225309, 220045, 223762, 225309, 220045, 220210, 223762, 220210, 223762, 220210, 220210, 223762, 220210, 220210, 223762, 220210, 225309, 220045, 220210, 223762, 225309, 220045, 223762, 225309, 220045, 220210, 223762, 220210, 223762, 220210, 220210, 223762, 220210, 220210, 223762, 220210],
        'value': ['120', '80', '95', '98', '120', '80', '95', '98', '96', '80', '95', '98', '96', '98', '96', '98', '96', 220210, 223762, 220210, 225309, 220045, 220210, 223762, 225309, 220045, 223762, 225309, 220045, 220210, 223762, 220210, 223762, 220210, 220210, 223762, 220210, 220210, 223762, 220210],
        'valuenum': [120, 80, 95, 98, 120, 80, 95, 120, 80, 95, 98, 95, 98, 95, 98, 95, 95, 98, 92, 93, 120, 80, 95, 98, 120, 80, 95, 120, 80, 95, 98, 95, 98, 95, 98, 95, 95, 98, 92, 93],
        'valueuom': ['mmHg', 'bpm', 'breaths/min', 'C', 'mmHg', 'bpm', 'breaths/min', 'C', 'C', 'bpm', 'breaths/min', 'C', 'C', 'bpm', 'breaths/min', 'C', 'C', 220210, 223762, 220210, 120, 80, 95, 98, 120, 80, 95, 120, 80, 95, 98, 95, 98, 95, 98, 95, 95, 98, 92, 93],
        'warning': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220210, 223762, 220210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    drgcodes = pd.DataFrame({
        'subject_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'drg_code': [870, 899, 878, 872, 675, 872, 786, 872, 872, 868, 872, 872, 872, 872, 812, 678, 453, 872, 870, 872, 870, 899, 878, 872, 675, 872, 786, 872, 872, 868, 872, 872, 872, 872, 812, 678, 453, 872, 870, 872],
    })
    return {
      'chartevents': chartevents,
      'drgcodes': drgcodes
    }

def test_process_features(mock_data):
    chartevents = mock_data['chartevents']
    csv_data = chartevents.to_csv('chartevents.csv', index=False)
    item_ids = {
        225309: "ART BP Systolic",
        220045: "HR",
        220210: "RR",
        223762: "Temperature C"
    }
    output_file = 'df.csv'

    result = process_features('chartevents.csv', output_file, item_ids)
    print(result)
    assert not result.empty
    assert set(result.columns) == set(['225309', '220045', '220210', '223762', 'subject_id'])

def test_add_diagnosis_column(mock_data):
    drgcodes_path = "drgcodes.csv"
    merged_data_path = "df.csv"
    output_path = "diagnosis_data.csv"
    mock_data['drgcodes'].to_csv(drgcodes_path, index=False)
    
    add_diagnosis_column(str(drgcodes_path), str(merged_data_path), str(output_path))
    
    result = pd.read_csv(output_path)
    assert not result.empty
    assert 'diagnosis' in result.columns
  
def test_impute_data(mock_data):
    input_path = "diagnosis_data.csv"
    output_path = "df_impute.csv"
    features = ['225309', '220045', '220210', '223762']
    impute_data(str(input_path), str(output_path), features)
    
    result = pd.read_csv(output_path)
    assert not result.empty
    assert '225309' in result.columns
    assert '220045' in result.columns
    assert '220210' in result.columns
    assert '223762' in result.columns

def test_prepare_and_save_data(mock_data):
    input_path = "df_impute.csv"
    train_data_path = "train_balanced_data.csv"
    test_data_path = "test_data.csv"
    features = ['225309', '220045', '220210', '223762']
    df_resampled, test_data = prepare_and_save_data(
        str(input_path), 0.3, 42, features, 'diagnosis', 
        str(train_data_path), str(test_data_path)
    )
    
    assert not df_resampled.empty
    assert not test_data.empty
    assert 'diagnosis' in df_resampled.columns
    assert 'diagnosis' in test_data.columns

def test_resample_test_val_data(mock_data):
    input_path = "test_data.csv"
    test_data_path = "test_resampled_data.csv"
    val_data_path = "val_resampled_data.csv"
    features = ['225309', '220045', '220210', '223762']
    resample_test_val_data(
        str(input_path), 0.5, 42, features, 'diagnosis', 
        str(test_data_path), str(val_data_path)
    )
    
    test_resampled = pd.read_csv(test_data_path)
    val_resampled = pd.read_csv(val_data_path)
    
    assert not test_resampled.empty
    assert not val_resampled.empty
    assert 'diagnosis' in test_resampled.columns
    assert 'diagnosis' in val_resampled.columns

def test_train_tabnet_model(mock_data):
    train_path = "train_balanced_data.csv"
    val_path = "val_resampled_data.csv"
    feature_importances_path = Path("feature_importances.txt")
    model_save_path = "tabnet_model"
    check_file_path = Path("tabnet_model.zip")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    optimizer_params = {"lr": 0.02}
    scheduler_params = {"milestones": [10, 20, 30], "gamma": 0.9}
    
    assert not train_data.empty, "Train data is empty"
    assert not val_data.empty, "Validation data is empty"
    
    train_tabnet_model(
        str(train_path), str(val_path), str(feature_importances_path), str(model_save_path),
        optimizer_params, scheduler_params, pretraining_lr=0.05, training_lr=0.05
    )
    
    assert check_file_path.exists(), f"Model file {check_file_path} does not exist"
    with open(feature_importances_path) as f:
        assert f.read()

def test_evaluate_tabnet_model(mock_data):
    model_path = "tabnet_model.zip"
    test_data_path = "test_resampled_data.csv"
    metrics_output_path = Path("metrics.txt")
  
    evaluate_tabnet_model(str(model_path), str(test_data_path), str(metrics_output_path))
    
    assert metrics_output_path.exists()
    with open(metrics_output_path) as f:
        metrics = f.read()
        assert "Accuracy" in metrics
        assert "Precision" in metrics
        assert "Recall" in metrics
        assert "F1-score" in metrics