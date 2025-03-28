#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

def plot_confusion_matrix(y_test, y_pred):
    ''' Plots a confusion matrix given test and predicted values '''
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_pred):
    ''' Plots a ROC curve given test and predicted values '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def evaluate_model(model, X_test, y_test):
    ''' Evaluates a model given test data with accuracy, balanced accuracy, classification report, ROC curve, AUC score, and confusion matrix '''
    print("Model Score: ", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_roc_curve(y_test, y_pred)
    print("AUC Score:", roc_auc_score(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

def don_model():
    """
    Don's Model
    """
    model = RandomForestClassifier()
    return model
    

def XGBoost_V1():
    '''
    XGBoost classifier
    '''
    # Create a model
    return xgb.XGBClassifier(random_state=42)

def XGBoost_V2():
    '''
    XGBoost classifier with hyperparameter tuning using GridSearchCV
    '''
    # Create the grid search estimator along with a parameter object containing the values to adjust.
    grid_tuned_model = xgb.XGBClassifier(random_state=42)

    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    grid_clf = GridSearchCV(grid_tuned_model, param_grid, verbose=3)
    return grid_clf

def XGBoost_V3():
    '''
    XGBoost classifier with hyperparameter tuning using RandomizedSearchCV
    '''
    # Create the parameter object for the randomized search estimator.    
    random_tuned_model = xgb.XGBClassifier(random_state=42)

    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    grid_clf = RandomizedSearchCV(random_tuned_model, param_grid, verbose=3)
    return grid_clf

def ADABoost_V1():
    '''
    ADABoost classifier
    '''
    # Create a model
    return AdaBoostClassifier(random_state=42)

def ADABoost_V2():
    '''
    ADABoost classifier with hyperparameter tuning using GridSearchCV
    '''
    # Create the grid search estimator along with a parameter object containing the values to adjust.
    grid_tuned_model = AdaBoostClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }
    grid_clf = GridSearchCV(grid_tuned_model, param_grid, verbose=3)
    return grid_clf

def ADABoost_V3():
    '''
    ADABoost classifier with hyperparameter tuning using RandomizedSearchCV
    '''
    # Create the parameter object for the randomized search estimator.
    random_tuned_model = AdaBoostClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }
    grid_clf = RandomizedSearchCV(random_tuned_model, param_grid, verbose=3)
    return grid_clf