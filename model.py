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
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def evaluate_model(model, X_test, y_test):
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
    return xgb.XGBClassifier()

def ADABoost_V1():
    '''
    ADABoost classifier
    '''
    # Create a model
    return AdaBoostClassifier(n_estimators=50, learning_rate=1)