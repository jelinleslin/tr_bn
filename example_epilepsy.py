from epilepsy_preprocess import preprocess_data
import pandas as pd
import numpy as np
from epilepsy_classifiers import MBCClassifier
import data_type
from export import get_adjacency_matrix_from_et
from utils import export_dsc

# ----------- Preprocess -------------#
# Load data
path = "data/"
# Read datasets
df_train, df_test, q_vars_after_merge, cut_list = preprocess_data(path)
df_variables = pd.DataFrame({"id":np.arange(df_train.shape[1]),"name":df_train.columns})
# Response variables
response = ["Engel.1st.yr","Engel.2nd.yr","Engel.5th.yr"]
# Get categories for all variables
data_train_t = data_type.data(df_train)
custom_classes = data_train_t.classes
# Combine UCSF and MNI datasets
df_all = df_train.append(df_test).reset_index().drop("index",axis=1)
df_all.iloc[:,:-3] = df_all.drop(response,axis=1).astype(np.float32)

# ----------- Train MBC -------------#
# Forbidden parents for the MBC
num_nds = df_all.shape[1]
forbidden_parent = [[] for _ in range(num_nds-3)]
forbidden_parent.append(range(num_nds-3) + [num_nds-1,num_nds-2])
forbidden_parent.append(range(num_nds-3) + [num_nds-1])
forbidden_parent.append(range(num_nds-3)) 
# Fit classifier
estimator = MBCClassifier(response =response, custom_classes=custom_classes, repeats = 20, metric_sem = "aic", metric_classifier= "aic", alpha = 2.5, forbidden_parents=forbidden_parent)
estimator.fit(df_all)

# ----------- Predict MBC -------------#
# Just as an example, we predict the same instances used for training the model
y_hat = estimator.predict_proba(df_all,repeats_inf=20)

# ----------- Predict MBC -------------#
# Get adjacency matrix of imputation model
adj_imputer = get_adjacency_matrix_from_et(estimator.et_sem)
# Get adjacency matrix of the first mbc
adj_mbc0 = get_adjacency_matrix_from_et(estimator.mbc_ets[0])
# Get adjacency matrix of the second mbc
adj_mbc1 = get_adjacency_matrix_from_et(estimator.mbc_ets[1])
# ... up to mbc_ets[19], given that repeats = 20 

# Export dsc files with the imputation model and the mbcs (Requires R package bnlearn)
path = "models/"
estimator.export_dsc(path)

