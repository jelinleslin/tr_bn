from discretize import fixed_freq_disc, fixed_freq_disc_test
import pandas
import copy

# Merge function for variables Automatisms.as.first.sz.manifestation (AUT) and Tonic.clonic as first.. (TC)
# AUT: (1Y, 2N)
# TC: (1Y, 2N)
#
# First seizure freedom manifestation: 1 AUT, 2 TC, 3 other 
def merge_fsm(data):    
    data_m = copy.deepcopy(data)
    aut_row = data_m.loc[:,"Automatisms.as.first.sz.manifestation..1Y.2N."]
    tc_row = data_m.loc[:,"Tonic.clonic.hyperkinetic.movements.as.first.sz.manifestation..1Y.2N."]
    fsm_row = [merge_fsm_aux(aut,tc) for aut,tc in zip(aut_row,tc_row)]
    data_m.loc[:,"Automatisms.as.first.sz.manifestation..1Y.2N."] =  fsm_row  
    colnames = list(data_m)    
    colnames[colnames.index("Automatisms.as.first.sz.manifestation..1Y.2N.")] = "First clinical manifestation during a seizure..1Automatisms2Tonic-clonic or hyperkinetic movements3Other"
    data_m.columns = colnames  
    colnames.remove("Tonic.clonic.hyperkinetic.movements.as.first.sz.manifestation..1Y.2N.")
    data_m = data_m.loc[:,colnames]
    return data_m
        
    

def merge_fsm_aux(aut,tc):
    if not pandas.isnull(aut):
        if aut == 1:
            return 1
    if pandas.isnull(tc):
        return tc
    if tc == 1:
        return 2
    if pandas.isnull(aut): 
        return aut
    return 3

# Preprocess epilepsy data
def preprocess_data(path):
    
    path_train = path + "dat_train.csv"
    # path_train = path + "dat_train_all.csv"
    data_train = pandas.read_csv(path_train)
    data_train = data_train.iloc[:,1:] # Remove row names
    path_test = path + "dat_test.csv"
    # path_test = path + "dat_test_all.csv"
    data_test = pandas.read_csv(path_test)
    data_test = data_test.iloc[:,1:] # Remove row names
    q_vars = [9,10,11,25,26,28,29,45,46]
    q_vars_after_merge = [9,10,11,24,25,27,28,44,45]
    freq = 30

    
    # Discretize variables
    _,  cut_list   =  fixed_freq_disc(data_train , q_vars, freq = freq)
    cut_list[2].remove(1.7)
    data_train_disc = fixed_freq_disc_test(data_train , q_vars, cut_list)
    data_test_disc = fixed_freq_disc_test(data_test , q_vars, cut_list)
    
    #Merge variables
    data_train_disc = merge_fsm(data_train_disc)
    data_test_disc = merge_fsm(data_test_disc) 
    
    
    path_train_prep = path + "dat_train_prep.csv"
    data_train_disc.to_csv(path_train_prep, index=False)
    path_test_prep = path + "dat_test_prep.csv"
    data_test_disc.to_csv(path_test_prep, index=False)
    return data_train_disc, data_test_disc, q_vars_after_merge, cut_list