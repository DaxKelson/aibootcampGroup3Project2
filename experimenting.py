#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import model as Group3Models
from math import sin, pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
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
    
    return random_search.best_estimator_
    
def model_logistic_regression_v1(X_train, y_train, SEED=42):
    """
        Logistic Regression
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],   # Regularization strength
        'solver': ['liblinear', 'lbfgs'] # Different solvers
    }

    model = LogisticRegression(max_iter=100)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def model_logistic_regression_v2(X_train, y_train, SEED=42):
    """
        Logistic Regression
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 200, 300],   # Regularization strength
        'solver': ['liblinear', 'lbfgs', 'cholesky'] # Different solvers
    }

    model = LogisticRegression(max_iter=200)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

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
    grid_clf = GridSearchCV(random_tuned_model, param_grid, verbose=3)
    return grid_clf
    return AdaBoostClassifier(n_estimators=50, learning_rate=1)

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

#Read in CSV
online_shopping_df = pd.read_csv('online_shoppers_intention.csv')

online_shopping_df.info()

online_shopping_df["VisitorType"].value_counts()
online_shopping_df["Month"].value_counts()

# online_shopping_df.describe()

months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
online_shopping_df["Month"] = online_shopping_df["Month"].map(months)

visitor = {'New_Visitor': 1, 'Returning_Visitor': 2, 'Other': 3}
online_shopping_df["VisitorType"] = online_shopping_df["VisitorType"].map(visitor)

plt.figure(figsize=(12, 6))
online_shopping_df.hist(figsize=(12, 8), bins=30, edgecolor="k")
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

print("\n🔹 Boxplots for Outlier Detection")
plt.figure(figsize=(12, 6))
sns.boxplot(data=online_shopping_df)
plt.xticks(rotation=90)
plt.show()

print("\n🔹 Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(online_shopping_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#Drop the target column, setup X and y datasets for train test split
X = online_shopping_df.drop('Revenue', axis=1)
y = online_shopping_df['Revenue']

# Set seed for reproducibility
SEED = 42

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
X_train.head()

#Encode VisitorType Column
# X_train['VisitorType'].value_counts()

# Initialize the OrdinalEncoder and MinMaxScaler
ordinal_encoder = OrdinalEncoder()
min_max_scaler = MinMaxScaler()

#Change Column to values created by encoder and scale them with the min max scaler
X_train['VisitorType'] = ordinal_encoder.fit_transform(X_train['VisitorType'].values.reshape(-1, 1))
X_train['VisitorType'] = min_max_scaler.fit_transform(X_train['VisitorType'].values.reshape(-1, 1))

ordinal_encoder = OrdinalEncoder()
min_max_scaler = MinMaxScaler()

X_train['Month'] = X_train['Month'].apply(lambda x: sin(x*(pi * 6)))

# Fit and transform the 'VisitorType' column with OrdinalEncoder
X_train['Month'] = ordinal_encoder.fit_transform(X_train['Month'].values.reshape(-1, 1))

# Now scale the encoded values to [0, 1] using MinMaxScaler
X_train['Month'] = min_max_scaler.fit_transform(X_train['Month'].values.reshape(-1, 1))

# Check the unique values after scaling
print(X_train['Month'].value_counts())

#SMOTEEN the X_train, y_train to balance the classes
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

smote = SMOTEENN(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
Group3Models.evaluate_models(X_train_bal, y_train_bal)