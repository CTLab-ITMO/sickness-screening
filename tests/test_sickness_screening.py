import pytest
import pandas as pd
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sickness_screening import (get_diagnoses_data, get_disease_info, get_analyzes_data,
                                combine_data, merge_and_get_data, balance_on_patients,
                                compress, choose, fill_values, train_model)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
mock_diagnoses = pd.DataFrame({
    'subject_id': [1, 2, 3],
    'icd_code': ['A01', 'A02', 'A03']
})

mock_all_diagnoses = pd.DataFrame({
    'icd_code': ['A01', 'A02', 'A03'],
    'long_title': ['Sepsis due to A01', 'cured', 'Sepsis due to A03']
})
mock_diagnoses_result = pd.DataFrame({
    'subject_id': [1, 2, 3],
    'long_title': ['Sepsis due to A01', 'cured', 'Sepsis due to A03']
})
mock_chartevents = pd.DataFrame({
    'subject_id': [1, 1, 2],
    'itemid': [220045, 220210, 223762],
    'charttime': ['2020-01-01', '2020-01-02', '2020-01-01'],
    'value': [120, 80, 95],
    'valuenum': [120.0, 80.0, 95.0],
    'valueuom': ['bpm', 'mmHg', 'bpm']
})

mock_blood = pd.DataFrame({
    'subject_id': [1, 2, 3],
    'itemid': [51222, 51279, 51240],
    'charttime': ['2020-01-01', '2020-01-02', '2020-01-01'],
    'value': [13.5, 4.5, 300],
    'valuenum': [13.5, 4.5, 300.0],
    'valueuom': ['g/dL', 'M/uL', 'K/uL']
})

mock_sepsis_info = pd.DataFrame({
    'subject_id': [1, 2, 3],
    'has_sepsis': [True, False, True]
})


@pytest.fixture
def mock_data():
    return {
        'mock_diagnoses': mock_diagnoses,
        'mock_all_diagnoses': mock_all_diagnoses,
        'mock_chartevents': mock_chartevents,
        'mock_blood': mock_blood,
        'mock_sepsis_info': mock_sepsis_info,
        'mock_diagnoses_result': mock_diagnoses_result
    }


def test_get_diagnoses_data(mock_data):
    result = get_diagnoses_data(patient_diagnoses_df=mock_data['mock_diagnoses'],
                                all_diagnoses_df=mock_data['mock_all_diagnoses'], title_column='long_title')
    assert not result.empty
    assert 'long_title' in result.columns


def test_get_disease_info(mock_data):
    result = get_disease_info(diagnoses_df=mock_data['mock_diagnoses_result'],
                              title_column='long_title', disease_str='sepsis',
                              subject_id_column='subject_id', log_stats=False)
    assert not result.empty
    assert 'has_sepsis' in result.columns


def test_get_analyzes_data(mock_data):
    itemids_map = {
        220045: 'Heart rate',
        220210: 'Respiratory rate',
        223762: 'Temperature Fahrenheit'
    }

    result = get_analyzes_data(analyzes_df=mock_data['mock_chartevents'],
                               subject_id_col='subject_id', itemid_col='itemid',
                               rest_columns=['subject_id', 'charttime', 'Heart rate', 'Respiratory rate',
                                             'Temperature Fahrenheit'],
                               itemids_map=itemids_map)
    assert not result.empty
    assert 'subject_id' in result.columns
    assert 'Heart rate' in result.columns
    assert 'Respiratory rate' in result.columns
    assert 'Temperature Fahrenheit' in result.columns


result_chartevents = None


def test_combine_data(mock_data):
    result = combine_data(first_data=mock_data['mock_diagnoses_result'],
                          second_data=mock_data['mock_chartevents'], log_stats=False)
    assert not result.empty
    assert 'subject_id' in result.columns


def test_merge_and_get_data(mock_data):
    result_comb = combine_data(first_data=mock_data['mock_diagnoses_result'],
                               second_data=mock_data['mock_chartevents'], log_stats=False)
    result = merge_and_get_data(merge_with_df=result_comb,
                                blood_df=mock_data['mock_blood'],
                                get_data_from_df=mock_data['mock_chartevents'],
                                disease_info_df=mock_data['mock_sepsis_info'], log_stats=False)
    assert not result.empty
    assert 'subject_id' in result.columns


def test_balance_on_patients(mock_data):
    itemids_map = {
        220045: 'Heart rate',
        220210: 'Respiratory rate',
        223762: 'Temperature Fahrenheit'
    }

    result_analyzes = get_analyzes_data(analyzes_df=mock_data['mock_chartevents'],
                                        subject_id_col='subject_id', itemid_col='itemid',
                                        rest_columns=['subject_id', 'charttime', 'Heart rate', 'Respiratory rate',
                                                      'Temperature Fahrenheit'],
                                        itemids_map=itemids_map)
    result_comb = combine_data(first_data=mock_data['mock_diagnoses_result'],
                               second_data=result_analyzes, log_stats=False)
    result_merge = merge_and_get_data(merge_with_df=result_comb,
                                      blood_df=mock_data['mock_blood'],
                                      get_data_from_df=mock_data['mock_chartevents'],
                                      disease_info_df=mock_data['mock_sepsis_info'], log_stats=False)
    result_balance = balance_on_patients(balancing_df=result_merge,
                                         disease_col='has_sepsis',
                                         subject_id_col='subject_id', number_of_patient_selected=1, filtering_on=1,
                                         log_stats=False)
    assert not result_balance.empty
    assert 'subject_id' in result_balance.columns


def test_compress(mock_data):
    result = compress(df_to_compress=mock_data['mock_chartevents'],
                      subject_id_col='subject_id')
    assert not result.empty
    assert 'subject_id' in result.columns


#
def test_choose(mock_data):
    itemids_map = {
        220045: 'Heart rate',
        220210: 'Respiratory rate',
        223762: 'Temperature Fahrenheit'
    }

    result_analyzes = get_analyzes_data(analyzes_df=mock_data['mock_chartevents'],
                                        subject_id_col='subject_id', itemid_col='itemid',
                                        rest_columns=['subject_id', 'charttime', 'Heart rate', 'Respiratory rate',
                                                      'Temperature Fahrenheit'],
                                        itemids_map=itemids_map)
    result_comb = combine_data(first_data=mock_data['mock_diagnoses_result'],
                               second_data=result_analyzes, log_stats=False)
    result_merge = merge_and_get_data(merge_with_df=result_comb,
                                      blood_df=mock_data['mock_blood'],
                                      get_data_from_df=mock_data['mock_chartevents'],
                                      disease_info_df=mock_data['mock_sepsis_info'], log_stats=False)
    result_balance = balance_on_patients(balancing_df=result_merge,
                                         disease_col='has_sepsis',
                                         subject_id_col='subject_id', number_of_patient_selected=1, filtering_on=1,
                                         log_stats=False)
    result = choose(compressed_df=result_balance,
                    has_disease_col='has_sepsis',
                    subject_id_col='subject_id')
    assert not result.empty
    assert 'subject_id' in result.columns


def test_fill_values(mock_data):
    itemids_map = {
        220045: 'Heart rate',
        220210: 'Respiratory rate',
        223762: 'Temperature Fahrenheit'
    }

    result_analyzes = get_analyzes_data(analyzes_df=mock_data['mock_chartevents'],
                                        subject_id_col='subject_id', itemid_col='itemid',
                                        rest_columns=['subject_id', 'charttime', 'Heart rate', 'Respiratory rate',
                                                      'Temperature Fahrenheit'],
                                        itemids_map=itemids_map)
    result_comb = combine_data(first_data=mock_data['mock_diagnoses_result'],
                               second_data=result_analyzes, log_stats=False)
    result_merge = merge_and_get_data(merge_with_df=result_comb,
                                      blood_df=mock_data['mock_blood'],
                                      get_data_from_df=mock_data['mock_chartevents'],
                                      disease_info_df=mock_data['mock_sepsis_info'], log_stats=False)
    result_balance = balance_on_patients(balancing_df=result_merge,
                                         disease_col='has_sepsis',
                                         subject_id_col='subject_id', number_of_patient_selected=1, filtering_on=1,
                                         log_stats=False)
    result_choosen = choose(compressed_df=result_balance,
                            has_disease_col='has_sepsis',
                            subject_id_col='subject_id')
    result = fill_values(balanced_df=result_choosen,
                         strategy='most_frequent')
    assert not result.empty
    assert 'subject_id' in result.columns


def test_train_model(mock_data):
    itemids_map = {
        220045: 'Heart rate',
        220210: 'Respiratory rate',
        223762: 'Temperature Fahrenheit'
    }

    result_analyzes = get_analyzes_data(analyzes_df=mock_data['mock_chartevents'],
                                        subject_id_col='subject_id', itemid_col='itemid',
                                        rest_columns=['subject_id', 'charttime', 'Heart rate', 'Respiratory rate',
                                                      'Temperature Fahrenheit'],
                                        itemids_map=itemids_map)
    result_comb = combine_data(first_data=mock_data['mock_diagnoses_result'],
                               second_data=result_analyzes, log_stats=False)
    result_merge = merge_and_get_data(merge_with_df=result_comb,
                                      blood_df=mock_data['mock_blood'],
                                      get_data_from_df=mock_data['mock_chartevents'],
                                      disease_info_df=mock_data['mock_sepsis_info'], log_stats=False)
    result_balance = balance_on_patients(balancing_df=result_merge,
                                         disease_col='has_sepsis',
                                         subject_id_col='subject_id', number_of_patient_selected=1, filtering_on=1,
                                         log_stats=False)
    result_choosen = choose(compressed_df=result_balance,
                            has_disease_col='has_sepsis',
                            subject_id_col='subject_id')
    result_filled = fill_values(balanced_df=result_choosen,
                                strategy='most_frequent')
    model = train_model(df_to_train=result_filled, categorical_cols=[],
                        columns_to_train_on=['Hemoglobin'])
    assert isinstance(model, RandomForestClassifier)
