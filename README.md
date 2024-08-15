# House Price Prediction Model
This repository contains a machine learning model that predicts house prices using the California Housing dataset. The project is designed to demonstrate key concepts in data preprocessing, exploratory data analysis (EDA), and model training with XGBoost.

# Project Overview
The goal of this project is to build a regression model that can accurately predict house prices based on various features provided in the California Housing dataset. The project walks through data loading, cleaning, visualization, and model training.

# Key Features
Data Handling: Efficiently manage and preprocess data using Pandas and NumPy.
Data Visualization: Gain insights through data visualization using Matplotlib and Seaborn.
Model Training: Train a regression model using the XGBoost algorithm.
Model Evaluation: Evaluate model performance using metrics from Scikit-Learn.

# Libraries Used
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
'''

# Steps Involved
Import Libraries: The necessary Python libraries for data manipulation, visualization, and model training are imported.
Load and Inspect Data: The California Housing dataset is loaded, converted into a Pandas DataFrame, and inspected to understand its structure.
Basic Data Checks: The dataset is checked for the number of rows and columns, missing values, and basic statistical measures.
Correlation and Visualization: A heatmap is created to understand the correlation between different features in the dataset.
Data Preparation: The data is prepared for model training by splitting into features (X) and the target variable (Y).

# How to Use
Clone the Repository:
'''
git clone https://github.com/Ayanda247/house-price-prediction.git
'''
Navigate to the project directory and install the necessary dependencies:
'''
pip install -r requirements.txt
'''
Run the Jupyter Notebook or Python script to train the model and evaluate its performance.

# Dataset
California Housing Dataset: This dataset includes various features such as median income, house age, average number of rooms, and more, across different districts in California.

# Results
The model's performance is evaluated using metrics such as mean squared error and R-squared score, providing insights into its predictive power.

# Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Any contributions are welcome!
