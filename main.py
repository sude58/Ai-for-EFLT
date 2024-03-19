import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn
from lazypredict import Supervised

# Excluding problematic regressors
Supervised.removed_regressors.append('QuantileRegressor')
Supervised.REGRESSORS.remove(('QuantileRegressor', sklearn.linear_model._quantile.QuantileRegressor))
Supervised.removed_regressors.append('RANSACRegressor')
Supervised.REGRESSORS.remove(('RANSACRegressor', sklearn.linear_model.RANSACRegressor))
Supervised.removed_regressors.append('Lars')
Supervised.REGRESSORS.remove(('Lars', sklearn.linear_model.Lars))
Supervised.removed_regressors.append('GammaRegressor')
Supervised.REGRESSORS.remove(('GammaRegressor', sklearn.linear_model.GammaRegressor))
Supervised.removed_regressors.append('PoissonRegressor')
Supervised.REGRESSORS.remove(('PoissonRegressor', sklearn.linear_model.PoissonRegressor))
 

# Constants
MODELS = {'e': ExtraTreesRegressor(), 'h': HistGradientBoostingRegressor(), 'x': XGBRegressor()}

def handle_data(data):
    data_unencoded = data.drop(columns=['leaid', 'achv', 'math', 'rla',
                        'LOCALE_VARS', 'DIST_FACTORS', 
                        'COUNTY_FACTORS', 'HEALTH_FACTORS'])

    data = pd.get_dummies(data_unencoded, columns=['leanm', 'grade', 'year', 'Locale4', 'Locale3', 'CT_EconType'], dtype=float)
    data.fillna(method='ffill', inplace=True)
    return data   

def subset_selection(data):
    while True:
        answer = input("Do you want to select subset? (Y/N)")
        if (answer == 'N'):
            return data
        elif (answer == 'Y'):
            print("Please read codebook for getting every possible selections?")
            subsets = input("Please enter subset to choose.").split()
            try:
                values = input("Please enter value to make subset.").split()
                subset = data.loc[data[subsets[0]] == float(values[0])]
            except:
                print("Error occured. Please retry.")
            else:
                return subset
        else:
            print("Invalid answer. Please type valid answer.")

def merge_subsets(subset1, subset2):
    return pd.concat([subset1, subset2])

# Not working
def feature_analysis(data, model):
    importance = None
    features = data.drop('achvz', axis=1)
    target = data['achvz']
    scaled_features = pd.DataFrame(normalize(features), columns=features.columns)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size = 0.2)
    model.fit(X_train, y_train)
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = permutation_importance(model, X_train, y_train, n_repeats=30).importances_mean
    
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    return importance_df
        

def trend_analysis(data):
    features = data.drop('achvz', axis=1)
    target = data['achvz']

    scaled_features = pd.DataFrame(normalize(features), columns=features.columns)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size = 0.2)

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)
    return models

def feature_visualization(data):
    data = data.head(20)
    
    plt.figure(figsize=(12, 18))
    plt.barh(data['Feature'], data['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    
    plt2 = plt
    plt2.show()
    return plt
    

def trend_visualization(models):
    model_names = list(models['Model'])
    r2_scores = list(models['R-Squared'])
    outlier = []
    outlier_score = []
    for model, score in zip(model_names, r2_scores):
        if score > 1 or score < -1:
            outlier.append(model)
            outlier_score.append(score)
    model_names = [x for x in model_names if x not in outlier]
    r2_scores = [x for x in r2_scores if x not in outlier_score]
    plt.figure(figsize=(18, 9))
    plt.barh(model_names, r2_scores, color='skyblue')
    plt.xlabel('R-Squared Score')
    plt.title('Model Performance Comparison')
    
    plt2 = plt
    plt2.show()
    return plt

def save_file(file, format, save_index=False):
    while True:
        options = input("Will you save file? (Y/N): ")
        if (options == 'N'):
            return
        elif (options == 'Y'):
            fileName = input("Enter name of file to save: ")
            break
        else:
            print("Invalid input. Please try again.")
    if (format == 'csv'):
        file.to_csv(f"{fileName}.csv", index=False)
    # elif (format == 'pdf'):
    #     file.savefig(file, format= 'pdf')

current_file = None
MODELS = {'e': ExtraTreesRegressor(), 'h': HistGradientBoostingRegressor(), 'x': XGBRegressor(), 'l': LGBMRegressor()}

while True:
    print("1. Process data\n2. Select subset\n3. Merge subsets\n4. Analyze trend\n5. Analyze important feature\n6. Visualize trend\n7. Visualize feature\n8. End program")
    options = input("Choose option to execute: ")
    if (int(options) == 1):
        try:
            fileName = input("Please enter name of file to process (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            processed_data = handle_data(file)
            current_file = processed_data
            save_file(processed_data, 'csv')
            
    elif (int(options) == 2):
        try:
            fileName = input("Please enter name of file to extract subset (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            subset = subset_selection(file)
            current_file = subset
            save_file(subset, 'csv')
    elif (int(options) == 3):
        try:
            fileName1 = input("Please enter name of file to extract subset (Leave empty if you just want to use current file): ")
            fileName2 = input("Please enter name of file to extract subset (Leave empty if you just want to use current file): ")
            if fileName1:
                file1 = pd.read_csv(fileName1)
            else:
                file1 = current_file
            if fileName2:
                file2 = pd.read_csv(fileName2)
            else:
                file2 = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            merged = merge_subsets(file1, file2)
            current_file = merged
            save_file(merged, 'csv')
    elif (int(options) == 4):
        try:
            fileName = input("Please enter name of file to analyze (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            models = trend_analysis(file)
            current_file = models
            save_file(models, 'csv', save_index=True)
    elif (int(options) == 5):
        try:
            fileName = input("Please enter name of data file to analyze (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
            model = input("Please enter name of model to analyze: ")
            if model in MODELS.keys():
                model = MODELS[model]
            else:
                raise ValueError()
        except ValueError:
            print("Can't find a model. Please enter existing model")
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            importance = feature_analysis(file, model)
            current_file = importance
            save_file(importance, 'csv')
    elif (int(options) == 6):
        try:
            fileName = input("Please enter name of file to visualize (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            plot = trend_visualization(file)
            current_file = plot
    elif (int(options) == 7):
        try:
            fileName = input("Please enter name of file to visualize (Leave empty if you just want to use current file): ")
            if fileName:
                file = pd.read_csv(fileName)
            else:
                file = current_file
        except:
            print("Can't find a file. Please enter existing filename")
        else:
            plot = feature_visualization(file)
            current_file = plot
    elif (int(options) == 8):
        break
    else:
        print("Invalid input. Please try again.")

