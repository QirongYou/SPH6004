import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import multilabel_confusion_matrix
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp/joblib' 
# Load the dataset
df = pd.read_csv('sph6004_assignment1_data_final.csv')

columns_to_use = [
    'bun_min', 'inr_max', 'gcs_verbal_0.0', 'ph_min', 'admission_age', 'Glucose_Variability', 'gcs_motor_6.0', 'aniongap_max', 'calcium_min', 'weight_admit', 'ph_max', 'glucose_max', 
    'BMI', 'sbp_min', 'albumin_min', 'Oxygenation_Index', 'Acid_Base_Balance_Indicator', 'baseexcess_max', 'dbp_min', 'aniongap_min', 'inr_min', 'ptt_max', 'lactate_min', 'spo2_min', 
    'bilirubin_total_min', 'calcium_max', 'height', 'potassium_max', 'hematocrit_max', 'potassium_max.1', 'temperature_max', 'glucose_max.2', 'sbp_max', 'glucose_min', 'gcs_motor_1.0', 
    'heart_rate_max', 'calcium_min.1', 'abs_lymphocytes_min', 'temperature_min', 'spo2_max', 'potassium_min.1'
]

# Extracting data and target
data = df[columns_to_use]
target = df.iloc[:, 0]
# Splitting the dataset into train and test sets
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, n_jobs=-1, verbose=2)

# Fitting Grid Search on the training data
grid_search.fit(train_x, train_y)

# Best parameters
print("Best parameters:", grid_search.best_params_)
# Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
# Predicting the test set results with the best estimator
predicted_y = grid_search.best_estimator_.predict(test_x)

# Calculating multilabel confusion matrix
confusion_matrix = multilabel_confusion_matrix(test_y, predicted_y)

# Displaying the confusion matrix
print(confusion_matrix)
# Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
# [[[4885 1899]
#   [1103 2297]]

#  [[7918  238]
#   [1939   89]]

#  [[4426 2596]
#   [1191 1971]]

#  [[8074  516]
#   [1016  578]]]
# 30238/40736=0.74
# Average Precision: 0.44487702295813863
# Average Recall: 0.42635582050404375
# Average F1 Score: 0.4051156385282203
