import plotly.express as px
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# 导入数据集
df = pd.read_csv('sph6004_assignment1_data_final.csv')

# 66 0.4675
#columns_to_use = ['gender','admission_age','heart_rate_min','heart_rate_max','sbp_min','sbp_max','dbp_min','dbp_max','mbp_max',
#                'temperature_min','temperature_max','spo2_min','spo2_max','spo2_mean','glucose_min','glucose_max','lactate_min',
#                  'ph_min','ph_max','baseexcess_max','hematocrit_max','calcium_min','calcium_max','glucose_max.1','potassium_max','sodium_min','sodium_max','platelets_min','albumin_min',
#                  'aniongap_min','aniongap_max','bun_min','calcium_min.1','glucose_max.2','sodium_min.1','potassium_min.1','potassium_max.1','abs_lymphocytes_min','abs_monocytes_min',
#                  'imm_granulocytes_min','fibrinogen_min','inr_min','inr_max','ptt_max','bilirubin_total_min','ck_cpk_min','ck_mb_min','height','weight_admit','BMI','Glucose_Variability',
#                  'Oxygenation_Index','Acid_Base_Balance_Indicator','gcs_min_4.0','gcs_min_8.0','gcs_min_9.0','gcs_min_10.0','gcs_min_11.0','gcs_min_12.0','gcs_min_14.0','gcs_motor_1.0',
#                  'gcs_motor_2.0','gcs_motor_5.167548641228663','gcs_motor_6.0','gcs_verbal_0.0','gcs_verbal_3.0']
# 50 0.4675
columns_to_use = [
    'bun_min', 'gcs_verbal_0.0', 'inr_max', 'ph_min', 'admission_age', 'aniongap_max', 'Glucose_Variability', 'gcs_motor_6.0', 'weight_admit', 'calcium_min', 
    'glucose_max', 'ph_max', 'BMI', 'sbp_min', 'Oxygenation_Index', 'albumin_min', 'Acid_Base_Balance_Indicator', 'dbp_min', 'inr_min', 'baseexcess_max', 'ptt_max', 
    'aniongap_min', 'lactate_min', 'spo2_min', 'bilirubin_total_min', 'calcium_max', 'potassium_max.1', 'height', 'potassium_max', 'gcs_motor_5.167548641228663', 
    'hematocrit_max', 'temperature_max', 'glucose_max.2', 'sbp_max', 'heart_rate_max', 'spo2_max', 'gcs_motor_1.0', 'glucose_min', 'calcium_min.1', 'abs_lymphocytes_min', 
    'temperature_min', 'platelets_min', 'sodium_min.1', 'gender', 'potassium_min.1', 'heart_rate_min', 'fibrinogen_min', 'glucose_max.1', 'mbp_max', 'spo2_mean'
]
# columns_to_use = [
#     'bun_min', 'inr_max', 'gcs_verbal_0.0', 'ph_min', 'admission_age', 'Glucose_Variability', 'gcs_motor_6.0', 'aniongap_max', 'calcium_min', 'weight_admit', 'ph_max', 
#     'glucose_max', 'BMI', 'sbp_min', 'albumin_min', 'Oxygenation_Index', 'Acid_Base_Balance_Indicator', 'baseexcess_max', 'dbp_min', 'aniongap_min', 'inr_min', 'ptt_max', 
#     'lactate_min', 'spo2_min', 'bilirubin_total_min', 'calcium_max', 'height', 'potassium_max', 'hematocrit_max', 'potassium_max.1'
# ]
data=df[columns_to_use]
target=df.iloc[:,0]
 
# 切分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data,target,test_size=0.2,random_state=42)

# xgboost模型初始化设置
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)
watchlist = [(dtrain,'train')]

# booster:
params={'booster':'gbtree',
        'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'auc',
        'max_depth':5,
        'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.025,
        'seed':0,
        'nthread':8,
        'gamma':0.15,
        'learning_rate' : 0.01}

# 建模与预测：100棵树
bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
y_pred = bst.predict(dtest)

# 设置阈值、评价指标

print ('Precision: %.4f' %metrics.precision_score(test_y,y_pred,average='micro'))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred,average='micro'))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred,average='micro'))
print ('Accuracy: %.4f' % metrics.accuracy_score(test_y,y_pred))

print("测试集每个样本的得分\n",y_pred)
ypred_leaf = bst.predict(dtest, pred_leaf=True)
print("测试集每棵树所属的节点数\n",ypred_leaf)
ypred_contribs = bst.predict(dtest, pred_contribs=True)
print("特征的重要性\n",ypred_contribs )

# Get feature names
feature_names = bst.feature_names


# Get feature importances
importances = bst.get_score(importance_type='gain')

# Convert importances to a DataFrame
importances_df = pd.DataFrame({
    'Feature': importances.keys(),
    'Importance': importances.values()
})

# Sort the DataFrame by importance
importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Select top 41 features
top_importances_df = importances_df.head(41)

# Plot using Plotly
fig = px.bar(top_importances_df, x='Importance', y='Feature', orientation='h', 
             title='Top 41 Feature Importances', 
             color='Importance', color_continuous_scale='Viridis')

# Save the plot

fig.show()
