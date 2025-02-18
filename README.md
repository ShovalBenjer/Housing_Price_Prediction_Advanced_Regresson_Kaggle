# Housing Price Prediction – Advanced Regression Techniques (Kaggle)
![image](https://github.com/user-attachments/assets/f0c089f6-1672-4dc9-bd2b-8020ea8e6261)


**A Comprehensive Solution for Predicting House Prices Using Advanced Regression, Feature Engineering, and Hyperparameter Tuning**

## Overview  
This repository provides a complete solution for predicting house prices using advanced regression techniques, dimensionality reduction, and hyperparameter tuning. Based on the Kaggle House Prices dataset, the goal is to accurately predict the final sale price of each home by leveraging sophisticated preprocessing, feature engineering, and model ensembling.
![image](https://github.com/user-attachments/assets/4160fb43-38fb-4dbf-a8a1-3bd09b994677)

## Table of Contents  
- [Overview](#overview)  
- [Key Steps](#key-steps)  
- [Models Used](#models-used)  
- [Project Methodology](#project-methodology)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results and Observations](#results-and-observations)  
- [Notebook Highlights](#notebook-highlights)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  
- [References](#references)  
- [Contact](#contact)

## Key Steps  
- **Data Preprocessing:**  
  Handling missing values, encoding categorical variables, and removing outliers.  

- **Feature Engineering:**  
  Adding new features, applying Box-Cox transformation to reduce skewness, and creating composite features.
  ![image](https://github.com/user-attachments/assets/f3e2bcd8-f8e3-4621-8e4f-0aadce108657)

- **Dimensionality Reduction:**  
  Applying Principal Component Analysis (PCA) to reduce feature space while retaining essential variance.  

- **Modeling:**  
  Training multiple regression models (Lasso, ElasticNet, Kernel Ridge, Gradient Boosting, XGBoost, LightGBM, KNN, Decision Tree, SVM, and LWLR).  
  ![image](https://github.com/user-attachments/assets/633824e9-9134-4ad3-bd7c-72f9a02b8de0)

- **Stacking & Averaging:**  
  Combining predictions from various base models using a meta-model (Lasso) for improved accuracy.  
  ![image](https://github.com/user-attachments/assets/4549060c-328e-420b-8f85-63da3301d622)

- **Hyperparameter Tuning:**  
  Optimizing key model parameters (e.g., for XGBoost) using Optuna.  
  ![image](https://github.com/user-attachments/assets/b10273ff-631a-49a2-acc8-cb689717e91b)

- **Evaluation:**  
  Using metrics such as RMSLE to assess model performance.

## Models Used  
- **Lasso & ElasticNet:** For robust linear regression and feature selection.  
- **Kernel Ridge Regression:** To capture non-linear relationships via the kernel trick.  
- **Gradient Boosting, XGBoost, LightGBM:** Ensemble methods for improved predictive performance.  
- **K-Nearest Neighbors (KNN):** A non-parametric method effective in minimizing RMSE.  
- **Decision Tree Regressor & SVR:** For interpretable predictions and capturing complex patterns.  
- **Locally Weighted Linear Regression (LWLR):** A non-parametric approach for local fitting.

## Project Methodology (Phase 4 Adaptation)  
### Objective  
Accurately predict housing prices by integrating advanced regression models with rigorous feature engineering and hyperparameter tuning.

### Research Questions  
1. Which regression model yields the lowest RMSLE on the housing dataset?  
2. How do transformations like Box-Cox and dimensionality reduction with PCA impact performance?  
3. Can hyperparameter tuning with Optuna further enhance model accuracy?

### Methodology  
- **Data Preprocessing:**  
  Fill missing values (median for numerical, mode for categorical), encode categories, and apply Box-Cox transformation on 'SalePrice'.  
- **Feature Engineering:**  
  Create additional features (e.g., total square footage, combined bathroom counts) to enrich the dataset.  
- **Dimensionality Reduction:**  
  Use PCA (retaining 95% variance) to simplify the feature space.  
- **Model Training:**  
  Train various regressors using cross-validation, then build stacked and averaged models with Lasso as the meta-model.  
- **Hyperparameter Tuning:**  
  Optimize XGBoost and other models using Optuna, evaluating performance with cross-validated RMSLE.  
- **Evaluation:**  
  Compare model performance using RMSE, MAE, R², Huber Loss, MAPE, and explained variance, and visualize results with boxplots and line graphs.

## Installation  
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShovalBenjer/Housing_Price_Prediction_Advanced_Regresson_Kaggle.git
   cd Housing_Price_Prediction_Advanced_Regresson_Kaggle
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Also install additional packages:
   ```bash
   pip install optuna scikit-optimize
   ```

## Usage  
1. **Prepare the Data:**  
   Ensure that the Kaggle House Prices `train.csv` and `test.csv` are in the repository root.
2. **Run the Main Script:**  
   ```bash
   python main.py
   ```
   The script will preprocess the data, train multiple models, perform stacking and hyperparameter tuning, and generate predictions in `submission.csv`.
3. **Explore the Notebook:**  
   Open the provided Jupyter notebook (detailed in the next section) for an in-depth view of the experimentation, including visualizations and Optuna tuning results.

## Results and Observations  
- The tuned KNN model achieved an RMSE of 0.2802, indicating strong predictive performance.
- Stacked and averaged models, using Lasso as a meta-model, further reduced prediction error.
- Feature engineering through PCA and Box-Cox transformation significantly enhanced model performance.
- Visualizations confirm the consistency of validation metrics across folds and the effectiveness of hyperparameter tuning.

## Notebook Highlights  
The detailed notebook includes:
- **Objective & Research Questions:**  
  A clear outline of the project goals and key questions.
- **Data Import and Setup:**  
  Installation of necessary packages (including Optuna) and data loading steps.
- **Preprocessing & Feature Engineering:**  
  Detailed code for missing value imputation, categorical encoding, Box-Cox transformation, and new feature creation.
- **Dimensionality Reduction:**  
  Application of PCA to scale down the feature set while preserving 95% of the variance.
- **Model Training & Evaluation:**  
  Implementation of various regression models with cross-validation, model ensembling via stacking, and evaluation using multiple metrics.
- **Hyperparameter Tuning:**  
  An objective function for Optuna is defined to optimize XGBoost parameters, along with visualization of optimization history and parameter importances.
- **Submission Generation:**  
  Final predictions are generated and saved to CSV files for both the stacked model and the optimized XGBoost model.
- **Visualization:**  
  Plots of validation metrics across folds and Optuna tuning graphs provide insights into model performance.

## Contributing  
Contributions are welcome! Please review the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to propose improvements, add new experiments, or refine documentation.

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements  
- **Kaggle** for providing the dataset.  
- The open-source community for the development of libraries such as scikit-learn, XGBoost, LightGBM, and Optuna.  
- Inspiration from various Kaggle kernels and academic research on advanced regression techniques.

## References  
1. [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
2. [Optuna Documentation](https://optuna.readthedocs.io/)  
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)  
4. [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
5. Box, G. E. P., & Cox, D. R. (1964). *An analysis of transformations.*  
6. Jolliffe, I. T. (2002). *Principal Component Analysis.*

## Contact  
For questions, feedback, or collaboration opportunities, please contact:

**Shoval Benjer**  
Creative Data Scientist | Tel Aviv - Jaffa, ISR  
GitHub: [ShovalBenjer](https://github.com/ShovalBenjer)  
Email: shovalb9@gmail.com  

---

**Enjoy exploring advanced regression techniques for housing price prediction, and feel free to contribute to this project!**
