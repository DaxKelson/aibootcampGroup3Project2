# Ecommerce Purchase Predictor

## Project Description

This project is an AI-driven E-commerce Purchase Predictor that determines whether a customer will make a purchase based on various behavioral and device-related features. The model leverages machine learning techniques to analyze user interaction data and predict purchasing likelihood.

The project includes a user-friendly GUI built with Tkinter, allowing users to input feature values dynamically and receive real-time predictions.

## Table of Contents

- [Motivation](#motivation)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Model Selection and Training](#model-selection-and-training)
- [Best Performing Model](#best-performing-model)
- [How to Install and Run](#how-to-install-and-run)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Motivation

E-commerce businesses often struggle with cart abandonment and customer retention. This project aims to provide a predictive solution to help businesses understand purchasing behavior and optimize their marketing efforts accordingly.

## Dataset and Preprocessing

- The dataset includes user interaction data such as time spent on a page, device type, operating system, and several other behavioral indicators.
  - ```Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q. https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset```
- We used `StandardScaler` to normalize continuous numerical features.
- Categorical features were encoded using `OneHotEncoder` and `OrdinalEncoder`.
- The data was split into training and testing sets to evaluate model performance effectively.

<img src="Slide Deck/Project2_Slide_5.jpg" alt="Cleanup and Preprocessing" width="200">

## Model Selection and Training

We experimented with multiple machine learning models, including:

1. **Logistic Regression**
2. **Decision Trees**
3. **Random Forest Classifier** (best performing)
4. **Gradient Boosting Machines (GBM)**

<img src="Slide Deck/Project2_Slide_9.jpg" alt="Model Performance Results" width="200">

### Best Performing Model: Random Forest Classifier

- **Balanced and Hyperparameter Tuned**: Grid search was used to optimize hyperparameters.
- **Performance Metrics:**
  - Accuracy: ~85%
  - Precision: 82%
  - Recall: 88%
- **Model was saved using `joblib` for deployment.**
<img src="Slide Deck/Project2_Slide_10.jpg" alt="Performance Metrics" width="200">

## How to Install and Run

### Prerequisites

- Python 3.8+
- Install dependencies using:
  ```sh
  pip install -r requirements.txt

# Sources
* https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost
* https://www.kaggle.com/code/bachnguyentfk/adaboost-hyperparameters-grid-search