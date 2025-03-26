#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import inspect

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def model_don_model(X_train, y_train, SEED=42):
    """
    Don's Model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model
    
def model_XGBoost_V1(X_train, y_train, SEED=42):
    '''
    XGBoost classifier
    '''
    # Create a model
    return xgb.XGBClassifier()

def model_ADABoost_V1(X_train, y_train, SEED=42):
    '''
    ADABoost classifier
    '''
    # Create a model
    return AdaBoostClassifier(n_estimators=50, learning_rate=1)

def evaluate_models(X_test, y_test):
    results = []
    for name, func in globals().items():
        if callable(func) and name.startswith("model_"):
            docstring = inspect.getdoc(func) or "No Comment"
            model = func(X_test, y_test)
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Description": docstring, "Accuracy": accuracy})
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("model_evaluation.csv", index=False)
    print("Evaluation saved to model_evaluation.csv")