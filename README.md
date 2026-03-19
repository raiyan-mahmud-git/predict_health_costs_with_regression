# predict_health_costs_with_regression

## Linear Regression Health Costs Calculator

This project uses a machine learning regression model to predict individual healthcare expenses based on demographic and lifestyle data. The model is built and trained using TensorFlow and Keras in Google Colaboratory.

# Objective

The goal of this project is to train a model that can accurately predict medical insurance costs (expenses) for individuals using features such as age, sex, BMI, number of children, smoking status, and region. The model must achieve a Mean Absolute Error (MAE) of less than 3500 to pass the challenge.

# Dataset

The dataset contains the following features:
age: Age of the individual
sex: Gender (male/female)
bmi: Body Mass Index
children: Number of dependents
smoker: Smoking status (yes/no)
region: Residential area
expenses: Medical costs (target variable)

# Data Preprocessing

Converted categorical variables into numerical format:
sex and smoker were mapped to binary values
region was transformed using one-hot encoding
Split the dataset into:
80% training data
20% testing data
Separated features and labels (expenses)
Applied normalization to improve model performance

# Model Architecture

A neural network was used for regression with the following structure:
Normalization layer
Dense layer (128 neurons, ReLU activation)
Dense layer (64 neurons, ReLU activation)
Output layer (1 neuron for prediction)

# The model was compiled using:

Optimizer: Adam
Loss function: Mean Absolute Error (MAE)
Metrics: MAE and Mean Squared Error (MSE)

# Training

The model was trained for multiple epochs with a validation split to monitor performance
Training was done on the training dataset, while validation helped prevent overfitting

# Evaluation

The model was evaluated on unseen test data.
Final result achieved a Mean Absolute Error (MAE) below 3500
This means predictions are, on average, within $3500 of actual healthcare costs

# Visualization

A scatter plot was generated comparing:
True values (actual expenses)
Predicted values
Points close to the diagonal line indicate accurate predictions.

# Conclusion

The model successfully learned patterns in the dataset and met the required performance threshold. This project demonstrates the application of regression techniques using neural networks for real-world cost prediction problems.

# Tools and Technologies

Python
TensorFlow / Keras
Pandas
NumPy
Matplotlib
Google Colaboratory

# Status
Completed and passed all required tests.
