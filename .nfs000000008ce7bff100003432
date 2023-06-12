from __future__ import print_function
import pandas
import sys
import numpy as np
from learn_structure_cache import hill_climbing_cache
from export import get_adjacency_matrix_from_et
from var_elim import PyFactorTree
from data_type import data as datat
import os


# Load Nltcs dataframe
d='nltcs'
for tw_bound in range(1, 10, 2):
  print(tw_bound)
  file_name = "data/{}_train_tw.csv".format(d)
  data = pandas.read_csv(file_name)
  data = data.replace(0,'no')
  data = data.replace(1,'yes')
  var_classes = [['yes','no'] for _ in range(len(data.columns))]
  # ----LEARNING BAYESIAN NETWORKS WITH BOUNDED TREEWIDTH---- #
  # Learn elimination tree (ET) with hc-et, using a tw bound of 3 and BIC as the objective score
  et = hill_climbing_cache(data, metric = 'bic', tw_bound = tw_bound, custom_classes=var_classes)
  # Learn ET with hc-et-poly, using a tw bound of 3 and BIC as the objective score
  et2 = hill_climbing_cache(data, metric = 'bic', tw_bound = tw_bound, custom_classes=var_classes, add_only=True)
  
  # Get adjacency matrix of the Bayesian network encoded by the ET et
  adj_mat = get_adjacency_matrix_from_et(et)
  print(adj_mat)
  a = np.array(adj_mat)
  mat = np.matrix(a)

          


  folder_name = "output_structure"
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
  
  file_path = os.path.join(folder_name, '{}_{}_structure.csv'.format(d, tw_bound))
  with open(file_path, 'wb') as f:
      for line in mat:
          np.savetxt(f, line, fmt='%d')
  
  print("File saved in the 'output_structure' folder.")

  