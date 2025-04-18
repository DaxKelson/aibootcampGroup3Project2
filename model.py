#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import os

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

def evaluate_model(model, name_and_iteration, description, X_train, X_test, y_train, y_test, print_plots: bool = False):
    ''' Evaluates a model given test data with accuracy, balanced accuracy, classification report, ROC curve, AUC score, and confusion matrix '''
    file_exists = os.path.isfile('gen_model_iterations_report.csv')
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    # Extract the metrics
    class_report = report['macro avg']
    #append to model_iterations file
    new_data = {
        'Name and Iterations' : name_and_iteration,
        'Description' : description,
        'Accuracy Score': accuracy_score(y_test, y_pred),
        'Balanced Accuracy Score': balanced_accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred),
        'Train Score': model.score(X_train, y_train),
        'Test Score': model.score(X_test, y_test),
        'Precision (Macro)': class_report['precision'],
        'Recall (Macro)': class_report['recall'],
        'F1-Score (Macro)': class_report['f1-score'],
        'Support (Macro)': class_report['support']
    }
    df = pd.DataFrame([new_data])
    df.to_csv('gen_model_iterations_report.csv', mode='a', header=not file_exists, index=False)  # 'a' for append mode, no header

    #print out scoring
    print("Model Score: ", model.score(X_test, y_test))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, y_pred))

    if(print_plots):
        plot_roc_curve(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)

    print("Large gap in score means overfitting: ")
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

def model_random_forest_model_V1(X_train, y_train, SEED=42):
    """
    Simple Random ForestModel with 100 n_estimators
    """
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def model_random_forest_model_V2(X_train, y_train, SEED=42):
    """
    Use RandomSearchCV to optimize the random forest model    
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=SEED)
    random_search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    random_search.fit(X_train, y_train)
    print("Best Parameters: ")
    print(random_search.best_params_)


    return random_search.best_estimator_
    
def model_logistic_regression_v1(X_train, y_train, SEED=42):
    """
        Logistic Regression
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],   # Regularization strength
        'solver': ['liblinear', 'lbfgs'] # Different solvers
    }

    model = LogisticRegression(max_iter=300)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters: ")
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def model_logistic_regression_v2(X_train, y_train, SEED=42):
    """
        Logistic Regression
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 200],   # Regularization strength
        'solver': ['liblinear', 'lbfgs'] # Different solvers
    }

    model = LogisticRegression(max_iter=500)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters: ")
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def model_xgboost_V1(X_train, y_train, SEED=42):
    '''
    XGBoost classifier with default parameters
    '''
    return xgb.XGBClassifier(random_state=42).fit(X_train, y_train)

def model_xgboost_V2(X_train, y_train, SEED=42):
    '''
    XGBoost classifier with hyperparameter tuning using GridSearchCV
    '''
    # Create the grid search estimator along with a parameter object containing the values to adjust.
    grid_tuned_model = xgb.XGBClassifier(random_state=42)

    # Create the parameter object for the grid search estimator.
    # The parameters are based on the XGBoost documentation and common practices.
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    # https://xgboost.readthedocs.io/en/latest/tutorials/model_tuning.html
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    grid_clf = GridSearchCV(grid_tuned_model, param_grid, verbose=3)
    grid_clf.fit(X_train, y_train)
    print("Best parameters found: ", grid_clf.best_params_)
    print("Best score found: ", grid_clf.best_score_)
    return grid_clf.best_estimator_

def model_xgboost_V3(X_train, y_train, SEED=42):
    '''
    XGBoost classifier with hyperparameter tuning using RandomizedSearchCV
    '''
    # Create the parameter object for the randomized search estimator.    
    random_tuned_model = xgb.XGBClassifier(random_state=42)

    # Create the parameter object for the grid search estimator.
    # The parameters are based on the XGBoost documentation and common practices.
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    # https://xgboost.readthedocs.io/en/latest/tutorials/model_tuning.html
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    rand_clf = RandomizedSearchCV(random_tuned_model, param_grid, verbose=3)
    rand_clf.fit(X_train, y_train)
    print("Best parameters found: ", rand_clf.best_params_)
    print("Best score found: ", rand_clf.best_score_)
    return rand_clf.best_estimator_

def model_adaboost_V1(X_train, y_train, SEED=42):
    '''
    ADABoost classifier
    '''
    # Create a model
    return AdaBoostClassifier(random_state=42).fit(X_train, y_train)

def model_adaboost_V2(X_train, y_train, SEED=42):
    '''
    ADABoost classifier with hyperparameter tuning using GridSearchCV
    '''
    # Create the grid search estimator along with a parameter object containing the values to adjust.
    grid_tuned_model = AdaBoostClassifier(random_state=42)

    # Create the parameter object for the grid search estimator.
    # The parameters are based on the adaboost documentation and common practices.
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }
    grid_clf = GridSearchCV(grid_tuned_model, param_grid, verbose=3)
    grid_clf.fit(X_train, y_train)
    print("Best parameters found: ", grid_clf.best_params_)
    print("Best score found: ", grid_clf.best_score_)
    return grid_clf.best_estimator_

def model_adaboost_V3(X_train, y_train, SEED=42):
    '''
    ADABoost classifier with hyperparameter tuning using RandomizedSearchCV
    '''
    # Create the parameter object for the randomized search estimator.
    random_tuned_model = AdaBoostClassifier(random_state=42)

    # Create the parameter object for the grid search estimator.
    # The parameters are based on the adaboost documentation and common practices.
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }
    rand_clf = RandomizedSearchCV(random_tuned_model, param_grid, verbose=3)
    rand_clf.fit(X_train, y_train)
    print("Best parameters found: ", rand_clf.best_params_)
    print("Best score found: ", rand_clf.best_score_)
    return rand_clf.best_estimator_

def evaluate_models(X_test, y_test):
     results = []
     for name, func in globals().items():
         if callable(func) and name.startswith("model_"):
            docstring = inspect.getdoc(func) or "No Comment"
            model = func(X_test, y_test)
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            f1_score = classification_report(y_test, y_pred)
            results.append({"Model": name, "Description": docstring, "Accuracy": accuracy, "f1 scores": f1_score})
     # Save results to CSV
     df = pd.DataFrame(results)
     df.to_csv("model_evaluation.csv", index=False)
     print("Evaluation saved to model_evaluation.csv")