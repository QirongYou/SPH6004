import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix

# Load the dataset
df = pd.read_csv('sph6004_assignment1_data_final.csv')

# Selecting the columns to use
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

# Creating and training the Random Forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(train_x, train_y)

# Predicting the test set results
predicted_y = model.predict(test_x)

# Calculating multilabel confusion matrix
confusion_matrix = multilabel_confusion_matrix(test_y, predicted_y)

# Displaying the confusion matrix
print(confusion_matrix)
# [[[4728 2056]
#   [1106 2294]]

#  [[7984  172]
#   [1956   72]]

#  [[4387 2635]
#   [1225 1937]]

#  [[8122  468]
#   [1044  550]]]
# 30074/40736=0.738
# Average Precision: 0.4465947824860341
# Average Recall: 0.4169599314712132
# Average F1 Score: 0.39435465182974605
