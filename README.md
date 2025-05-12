# Crab Age Prediction Model

## Overview

This repository contains a machine learning solution for predicting the age of crabs based on physical measurements. The model utilizes an ensemble approach combining multiple regression algorithms to achieve optimal performance. This solution was developed for a machine learning competition where the goal is to predict crab age from a dataset generated from a deep learning model trained on the original Crab Age Prediction dataset.

## Features

- Comprehensive data exploration and visualization
- Advanced feature engineering
- Model comparison and selection
- Hyperparameter optimization
- Ensemble learning approach
- Detailed performance evaluation

## Requirements

The following Python libraries are required:

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.4.0
lightgbm>=3.2.0
```

You can install all requirements using:

```bash
pip install -r requirements.txt
```

## Dataset

The model requires two CSV files:

1. `train.csv` - Training data with features and target variable (Age)
2. `test.csv` - Test data for prediction

Optionally, you can include:
- `original_crab_dataset.csv` - The original crab age dataset (if available)

## Features Used

The model uses the following features and derived features:

### Original Features:
- Sex (categorical)
- Length
- Diameter
- Height
- Weight
- Shucked Weight
- Viscera Weight
- Shell Weight

### Engineered Features:
- Volume (Length × Diameter × Height)
- Density (Weight / Volume)
- Shell Ratio (Shell Weight / Weight)
- Meat Weight (Weight - Shell Weight)
- Weight Ratios (various weight proportions)
- Feature interactions

## Model Architecture

The solution employs an ensemble of multiple regression models:

1. Gradient Boosting Regressor
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. ElasticNet Regressor

The final prediction is an average of the top-performing models, which helps reduce overfitting and improves generalization.

## Usage

1. Clone this repository:
```bash
git clone https://github.com/vishwasbhairab/crab-age-prediction.git
cd crab-age-prediction
```

2. Place your dataset files in the repository root:
   - `train.csv`
   - `test.csv`
   - `original_crab_dataset.csv` (optional)

3. Run the model:
```bash
python crab_age_model.py
```

4. The script will:
   - Perform data exploration and create visualization files
   - Train and evaluate multiple models
   - Select and tune the best models
   - Generate an ensemble prediction
   - Create a submission file named `crab_age_predictions.csv`

## Output Files

The script generates several output files:

1. `crab_age_predictions.csv` - The submission file with predicted ages
2. Visualization files:
   - `feature_distributions.png` - Histograms of feature distributions
   - `correlation_matrix.png` - Correlation between features
   - `age_distribution.png` - Distribution of target variable
   - `sex_distribution.png` - Distribution of crab sex
   - `feature_importance.png` - Importance of each feature (if applicable)

## Model Performance

The model's performance is evaluated using Root Mean Squared Error (RMSE), which is displayed during training. The cross-validation approach helps ensure that the performance estimate is reliable.

## Customization

You can modify the hyperparameter search space in the script to explore different model configurations. The ensemble approach can also be adjusted to include different combinations of models.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This solution was inspired by various regression techniques and ensemble methods
- Special thanks to the competition organizers for providing the dataset
