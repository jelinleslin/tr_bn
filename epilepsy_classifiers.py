#--------- Test learning methods -----------#
from var_elim import PyFactorTree
from export import get_adjacency_matrix_from_et
from mbc import learn_mbc_generative
from copy import deepcopy
import numpy as np
import pandas as pd
import data_type
from sem_soft import tsem_cache
from utils import multi_sample_dataset, export_dsc
from pathlib import Path

class MBCClassifier(object):
    """ MBC classifier with imputation
    """
    def __init__(self, response, custom_classes, metric_sem="aic", metric_classifier="aic", alpha=1.0,forbidden_parents=None, repeats=5):
        self.response = response
        self.repeats = repeats

        self.mbc_ets = []
        self.mbc_params = []
        self.estimators = []
        self.custom_classes = custom_classes
        self.imputer = None
        self.metric_sem = metric_sem
        self.metric_classifier = metric_classifier
        self.alpha = alpha
        self.forbidden_parents = deepcopy(forbidden_parents)

    def fit(self,df,random_state=1):
        # Get index of the response columns
        cll_query = [np.where(ri==df.columns)[0][0] for ri in self.response]
        # Train imputer
        # SEM + multiple imputation
        et_sem, _, fiti_sem, df_complete, _ = tsem_cache(df,custom_classes=self.custom_classes, metric=self.metric_sem, complete_prior="random")
        self.imputer = fiti_sem
        self.et_sem = et_sem
        df_imputed_list = multi_sample_dataset(self.imputer,df,self.custom_classes, random_state=random_state, repeats=self.repeats)
        self.df_imputed_list = df_imputed_list 
         

        # Fit a model to each dataset
        for dfi in df_imputed_list:
            et_mbc = learn_mbc_generative(dfi, cll_query, pruned=False, et0=None, u=5, forbidden_parent=deepcopy(self.forbidden_parents), metric=self.metric_classifier, custom_classes=self.custom_classes)

            # Get factor tree
            fiti_mbc = PyFactorTree([et_mbc.nodes[j].parent_et for j in range(et_mbc.nodes.num_nds)], [et_mbc.nodes[j].nFactor for j in range(et_mbc.nodes.num_nds)], [[j]+et_mbc.nodes[j].parents.display() for j in range(et_mbc.nodes.num_nds)], [len(c) for c in self.custom_classes])
            df_imputed_t = data_type.data(dfi,classes = self.custom_classes)  
            fiti_mbc.learn_parameters(df_imputed_t, alpha = self.alpha)
            self.estimators.append(fiti_mbc)
            self.mbc_ets.append(et_mbc)
            self.mbc_params.append(fiti_mbc.get_parameters())
    
    def predict_proba(self, df,random_state=1,repeats_inf=30):
        # Impute pred data
        num_cols = len(self.custom_classes)
        df_inf = df.copy()
        df_inf[self.response] = np.nan
        df_imputed_list = multi_sample_dataset(self.imputer,df_inf,self.custom_classes, random_state=random_state, repeats=repeats_inf)
        y_hat = np.zeros([df.shape[0],len(self.response)])
        df_imputed = pd.concat(df_imputed_list, axis=0)
        df_imputed_t = data_type.data(df_imputed,classes = self.custom_classes) 
        # Get prediction for each class
        for estimator in self.estimators:
            # Make predictions
            y_hat_new = estimator.pred_data(df_imputed_t, [0,0,0], range(num_cols-3,num_cols), range(0,num_cols-3))
            # Average over responses
            y_hat_new = y_hat_new.reshape([repeats_inf,df.shape[0],3])
            y_hat_avg = np.mean(y_hat_new,axis=0)
            # Sum response
            y_hat = y_hat + y_hat_avg
        y_hat = y_hat/len(self.estimators)
        return y_hat

    def export_dsc(self,path):
        try:
            Path(path).mkdir(parents=True)
        except:
            print "Warning: Could not create folder '{}'".format(path)
        # Export imputer
        export_dsc(self.et_sem,self.imputer,self.custom_classes,path+"imputer.dsc")
        # Export mbcs
        for i, (mbc_et, estimator) in enumerate(zip(self.mbc_ets,self.estimators)):
            export_dsc(mbc_et,estimator,self.custom_classes,path+"estimator_{}.dsc".format(i))


