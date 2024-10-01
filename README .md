# Random Forest Time Series Forecasting

This project aims to predict future values of a time series using a Random Forest Regressor model. The dataset used contains historical sales data, and the goal is to forecast future sales values.

## Project Structure

- `data_url`: Path to the dataset file.
- `train_set` and `test_set`: Data is split into training and test sets.
- `MinMaxScaler`: Used to scale features and targets.
- `RandomForestRegressor`: Model used for prediction.
- `GridSearchCV`: Used for hyperparameter tuning.

## Dataset

The dataset file used in this example is an Excel file with the following columns:
- `Gün` (Date column)
- `Net Satış Miktarı_MA` (Target column)

## Installation

Make sure to have the required libraries installed. You can install them using pip:

```
pip install pandas numpy matplotlib scikit-learn openpyxl
```

## Usage
Load Data:
Update the data_url variable with the path to your dataset file.

Prepare Data:
The script prepares the data by scaling features and targets, and creating lagged features for time series prediction.

Train Model:
The Random Forest Regressor model is trained using GridSearchCV to find the best hyperparameters.

Evaluate Model:
Model performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared, and accuracy metrics.

Forecast Future Values:
The model forecasts future values based on the test set.

Save Predictions:
The forecasted future values are saved to an Excel file.

## Results
The script will generate plots showing the training data, actual values, and Random Forest predictions.
Future predictions will be visualized and saved to an Excel file.
Example Output
Plots:

Training data vs. Actual values vs. Random Forest Predictions
Historical data vs. Future Predictions
Excel File:

The future predictions are saved in an Excel file at the specified location.
## Notes
Ensure that the dataset file and paths are correctly specified.
Adjust the model parameters and hyperparameters as needed to improve performance.
