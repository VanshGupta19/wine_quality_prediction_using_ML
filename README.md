# Wine Quality Prediction Using Machine Learning

End-to-end predictive model (regression) built for the Predictive Analytics course (CA3).

## Problem statement
Predict the sensory quality score (0-10) of Portuguese *vinho verde* wines from 11 physicochemical properties, and identify which properties influence quality most.

## Dataset
UCI Wine Quality dataset (Cortez et al., 2009): https://archive.ics.uci.edu/dataset/186/wine+quality  
- `data/winequality-red.csv` - 1,599 wines  
- `data/winequality-white.csv` - 4,898 wines  
- 11 input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) + target `quality`. No missing values. License: CC BY 4.0.

## Methodology (KDD stages)
1. **Problem definition & data selection** - combine red and white wines, add a wine type column.
2. **Data understanding** - distributions, class imbalance, correlation matrix, boxplots.
3. **Preprocessing** - remove 1,177 duplicate rows (5,320 remain), encode wine type, 80/20 stratified train/test split, cap extreme outliers (3×IQR, limits learned on train only), standard scaling inside a Pipeline.
4. **Feature selection** - Random Forest importance and mutual information; all 12 features kept.
5. **Modelling** - Dummy baseline, Linear, Ridge, Decision Tree, SVR, Random Forest, Gradient Boosting, XGBoost, compared with 5-fold cross-validation.
6. **Tuning** - RandomizedSearchCV on Random Forest and XGBoost.
7. **Evaluation** - MAE, MSE, RMSE, R² on the unseen test set, plus rounded-score accuracy.
8. **Interpretation** - permutation importance, actual-vs-predicted and residual plots.

## Results (test set, final tuned XGBoost)
| Metric | Value |
|---|---|
| MAE | 0.524 |
| MSE | 0.460 |
| RMSE | 0.678 |
| R² | 0.407 |
| Within ±1 quality point | 97.3% |

Mean-baseline MAE is 0.695 and linear regression R² is 0.313. Alcohol is the most important feature, followed by free sulfur dioxide and volatile acidity.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook wine_quality_prediction.ipynb
```
Run from the repository root so the `data/` folder is found.

## Files
- `wine_quality_prediction.ipynb` - full implementation with outputs
- `data/` - dataset files
- `requirements.txt`

## Reference
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*. DOI: 10.24432/C56S3T
