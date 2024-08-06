# Housing_Price_Prediction_Advanced_Regresson_Kaggle

This repository contains a comprehensive solution for predicting house prices using advanced regression techniques, dimensionality reduction, and hyperparameter tuning. The dataset used is from the Kaggle House Prices competition. The goal is to predict the final price of each home based on a variety of features.

This project includes the following key steps:

Data Preprocessing: Handling missing values, transforming categorical variables, and removing outliers.
Feature Engineering: Adding new features and applying Box-Cox transformation to reduce skewness.
Dimensionality Reduction: Using Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.
Modeling: Implementing multiple regression models including Lasso, ElasticNet, Kernel Ridge Regression, Gradient Boosting, XGBoost, LightGBM, K-Nearest Neighbors, Decision Tree Regressor, and Support Vector Machine.
Stacking and Averaging Models: Combining the predictions of multiple models to improve accuracy.
Hyperparameter Tuning: Using Optuna for tuning the hyperparameters of XGBoost to find the best model configuration.
Evaluation: Calculating Root Mean Squared Logarithmic Error (RMSLE) to evaluate model performance.
Models Used
Lasso: Useful for feature selection by enforcing sparsity.
ElasticNet: Combines the properties of Lasso and Ridge regression.
Kernel Ridge Regression: Applies ridge regression in a higher-dimensional space using the kernel trick.
Gradient Boosting Regressor: Builds an ensemble of trees where each tree corrects the errors of the previous ones.
XGBoost: Highly efficient and flexible gradient boosting framework.
LightGBM: Efficient gradient boosting framework with high scalability.
K-Nearest Neighbors (KNN): Predicts values based on the closest neighbors in the training set.
Decision Tree Regressor: Splits data into subsets based on feature values.
Support Vector Machine (SVM): Uses support vectors and margins for regression with a linear kernel.

How to Use
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/house-prices-prediction.git
cd house-prices-prediction
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Script:
Ensure the train and test datasets are in the root directory and run the script:

bash
Copy code
python main.py
Output:
The script will preprocess the data, train multiple models, and output the predictions to a CSV file named submission.csv.

Results
The results of the model training and evaluation are printed in the console. The best model configuration found using Optuna is also printed, along with the corresponding RMSLE score.

Conclusion
This repository demonstrates the use of advanced regression techniques, PCA for dimensionality reduction, and hyperparameter tuning with Optuna to effectively predict house prices. It combines multiple models using stacking and averaging to achieve the best results.

License
This project is licensed under the MIT License.

Acknowledgements
Kaggle for providing the dataset.
The open-source community for developing the libraries used in this project.
Feel free to contribute, open issues, and suggest improvements!
