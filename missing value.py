import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('sph6004_assignment1_data.csv')

# For mean imputation
df['height'].fillna(df['height'].mean(), inplace=True)
df['heart_rate_mean'].fillna(df['heart_rate_mean'].mean(), inplace=True)
df['resp_rate_mean'].fillna(df['resp_rate_mean'].mean(), inplace=True)
df['temperature_mean'].fillna(df['temperature_mean'].mean(), inplace=True)
df['spo2_mean'].fillna(df['spo2_mean'].mean(), inplace=True)
df['temperature_min.1'].fillna(df['temperature_min.1'].mean(), inplace=True)
df['temperature_max.1'].fillna(df['temperature_max.1'].mean(), inplace=True)

# For median imputation
df['weight_admit'].fillna(df['weight_admit'].median(), inplace=True)
df['heart_rate_min'].fillna(df['heart_rate_min'].median(), inplace=True)
df['heart_rate_max'].fillna(df['heart_rate_max'].median(), inplace=True)
df['resp_rate_min'].fillna(df['resp_rate_min'].median(), inplace=True)
df['resp_rate_max'].fillna(df['resp_rate_max'].median(), inplace=True)
df['temperature_min'].fillna(df['temperature_min'].median(), inplace=True)
df['temperature_max'].fillna(df['temperature_max'].median(), inplace=True)
df['spo2_min'].fillna(df['spo2_min'].median(), inplace=True)
df['spo2_max'].fillna(df['spo2_max'].median(), inplace=True)

imputer = KNNImputer(n_neighbors=5)

# Columns to impute
columns_to_impute = ['glucose_min', 'glucose_max', 'glucose_mean', 
                    'lactate_min', 'lactate_max', 'ph_min', 'ph_max', 
                    'so2_min', 'so2_max', 'po2_min', 'po2_max', 'pco2_min',
                    'pco2_max', 'aado2_min', 'aado2_max', 'aado2_calc_min',
                    'aado2_calc_max', 'pao2fio2ratio_min', 'pao2fio2ratio_max',
                    'baseexcess_min', 'baseexcess_max', 'bicarbonate_min',
                    'bicarbonate_max', 'totalco2_min', 'totalco2_max', 'hematocrit_min',
                    'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max',
                    'carboxyhemoglobin_min', 'carboxyhemoglobin_max',
                    'methemoglobin_min', 'methemoglobin_max',
                    'chloride_min', 'chloride_max', 'calcium_min',
                    'calcium_max', 'glucose_min.1', 'glucose_max.1',
                    'potassium_min', 'potassium_max', 'sodium_min',
                    'sodium_max', 'hematocrit_min.1', 'hematocrit_max.1',
                    'hemoglobin_min.1', 'hemoglobin_max.1', 'platelets_min',
                    'platelets_max', 'wbc_min', 'wbc_max', 'albumin_min', 'albumin_max',
                    'globulin_min', 'globulin_max', 'total_protein_min', 'total_protein_max',
                    'aniongap_min', 'aniongap_max', 'bicarbonate_min.1', 'bicarbonate_max.1',
                    'bun_min', 'bun_max', 'calcium_min.1', 'calcium_max.1',
                    'chloride_min.1', 'chloride_max.1', 'glucose_min.2', 'glucose_max.2',
                    'sodium_min.1', 'sodium_max.1', 'potassium_min.1', 'potassium_max.1',
                    'abs_basophils_min', 'abs_basophils_max', 'abs_eosinophils_min',
                    'abs_eosinophils_max', 'abs_lymphocytes_min', 'abs_lymphocytes_max',
                    'abs_monocytes_min', 'abs_monocytes_max', 'abs_neutrophils_min',
                    'abs_neutrophils_max', 'atyps_min', 'atyps_max', 'bands_min', 'bands_max',
                    'imm_granulocytes_min', 'imm_granulocytes_max', 'metas_min', 'metas_max',
                    'nrbc_min', 'nrbc_max', 'd_dimer_min', 'd_dimer_max',
                    'fibrinogen_min', 'fibrinogen_max', 'thrombin_min', 'thrombin_max',
                    'inr_min', 'inr_max', 'pt_min', 'pt_max', 'ptt_min', 'ptt_max',
                    'alt_min', 'alt_max', 'alp_min', 'alp_max', 'ast_min', 'ast_max',
                    'amylase_min', 'amylase_max', 'bilirubin_total_min', 'bilirubin_total_max',
                    'bilirubin_direct_min', 'bilirubin_direct_max', 'bilirubin_indirect_min', 'bilirubin_indirect_max',
                    'ck_cpk_min', 'ck_cpk_max', 'ck_mb_min', 'ck_mb_max', 'ggt_min', 'ggt_max',
                    'ld_ldh_min', 'ld_ldh_max']

df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Assuming df is your DataFrame
imp = IterativeImputer(max_iter=10, random_state=0)

# Columns to impute
columns_to_impute = [ 'mbp_min', 'mbp_max', 'mbp_mean', 'sbp_min', 'sbp_max', 'sbp_mean',
                    'dbp_min', 'dbp_max', 'dbp_mean', 'gcs_min', 'gcs_motor', 'gcs_verbal',
                    'gcs_eyes', 'gcs_unable']

df[columns_to_impute] = imp.fit_transform(df[columns_to_impute])
df.to_csv('sph6004_assignment1_data_edited.csv', index=False)