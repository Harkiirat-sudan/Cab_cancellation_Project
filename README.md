# Cab Cancellation Prediction Project

## Overview
This project aims to develop a machine learning model that predicts whether a cab booking will be canceled. By predicting cancellations in advance, cab service companies can reduce operational disruptions, improve customer satisfaction, and allocate resources more efficiently.

## Use Cases
- **Operational Efficiency:** Predict high-probability cancellation bookings and avoid dispatching vehicles unnecessarily.
- **Resource Allocation:** Ensure drivers are assigned to confirmed rides, minimizing downtime and losses.
- **Customer Experience:** Prevent situations where passengers cancel close to the trip start time, which usually causes inconvenience to both drivers and other potential passengers.

## Conclusions Drawn
- **Addressing Imbalance:** The data exhibited class imbalances. Using the **SMOTE (Synthetic Minority Over-sampling Technique)** effectively balanced the instances of cab cancellations before training.
- **Best Model:** While a standard Decision Tree showed signs of overfitting, the **Pruned Decision Tree** (`max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5`) offered a robust, balanced performance on both training and test datasets.
- **Performance Evaluation:** Both Logistic Regression and Decision Tree models were evaluated using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC. Confusion matrices and classification reports were generated to visualize the results.
- **Key Features:** Features influencing cancellations included `booking_lead_time` (difference between booking and trip start time) and geographical destination coordinates (`to_lat`, `to_long`).

## How to Run

1. **Prerequisites:** 
   Ensure you have Python installed, along with the required libraries. You can install them using `pip`:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
   ```

2. **Dataset:** 
   Ensure that the dataset file `YourCabs.csv` is located in the same directory as the Python script.

3. **Execution:** 
   Run the Python script from your terminal:
   ```bash
   python "cabcancellation(hk).py"
   ```
   Alternatively, you can open and run the Jupyter Notebook (`CabCancellation(Hk).ipynb`) for an interactive step-by-step review of the data visualization and model building.

4. **Outputs:** 
   The script performs data cleaning, handles missing values, detects and removes outliers, and trains the machine learning models. Finally, the resulting models will be saved to your local directory (`logistic_regression_model.pkl` and `decision_tree_model.pkl`) for future use.
