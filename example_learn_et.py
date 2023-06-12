from __future__ import print_function
import pandas
import sys
import numpy as np
from learn_structure_cache import hill_climbing_cache
from export import get_adjacency_matrix_from_et
from var_elim import PyFactorTree
from data_type import data as datat


# Load ASIA dataframe
d='nltcs'
for tw_bound in range(1, 4, 2):
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
  with open('{}_{}_structure.csv'.format(d,tw_bound),'wb') as f:
      for line in mat:
          np.savetxt(f, line, fmt='%d')
  # ----PARAMETER LEARNING---- #
  num_vars = len(var_classes) #Number of variables
  
  # Get cppet from from ET
  et_descriptor = [[et.nodes[i].parent_et for i in range(num_vars)], [et.nodes[i].nFactor for i in range(num_vars)], [[i] + et.nodes[i].parents.display() for i in range(num_vars)], [len(c) for c in var_classes]]
  cppet = PyFactorTree(et_descriptor[0], et_descriptor[1], et_descriptor[2],et_descriptor[3])
  # Transfrom dataframe to data_type
  data_p = datat(data,var_classes) 
    # Learn parameters: alpha is the Dirichlet hyperparameter for the Bayesian estimation. 
  #                   If alpha=0, the Maximum likelihood parameters are obtained 
  cppet.learn_parameters(data_p, alpha = 1)
  #print(cppet)
  original_stdout = sys.stdout
  #with open('asia.net','w') as f:
  	 #sys.stdout = f
  #print('nothing')
  #print('net')
  #print('{')
  #print('}')
  #for node in range(num_vars):
  #   print ('node x'+str(node))
  #   print ('{')
  #   print ('states = ( "0" "1" );')
  #   print ('}')
  
  #def print_cpt_net(v):
  #    if (len(v)==2):
  #       return( '('+ str(v[0])+ ' ' + str(v[1]) + ')') 
  #    v1 = v[0:(int(len(v)/2))];
  #    #print(v1)
  #    v2 = v[int(len(v)/2):len(v)];
  #    #print(v2)
  #    return( '(' + print_cpt_net(v1) + print_cpt_net(v2) +')')
     
  
  #for node in range (num_vars):
  #   parents = et.nodes[node].parents.display()
  #   print ('potential ( x' + str(node)+  ' ' + '|' + ' '  , end ='')
  #   for parent in parents:
  #      print ('x'+str(parent)+ ' ' , end ='')
  #   factor = cppet.get_factor(num_vars + node)
  #   parameters = factor.get_prob()
  #   #print ([[et.nodes[i].parent_et for i in range(num_vars)], [et.nodes[i].nFactor for i in range(num_vars)], [[i] + et.nodes[i].parents.display() for i in range(num_vars)]])
  #   print (')''\n''{')
  #   print('data ='+' '  +str(print_cpt_net(parameters) + ';'))
  #   print ('}')
     
     #print('g') if (len(parents) > 0)
     #for parent in parents:
       # print ('x'+str(parent))
  	        #print ('Potential x'+str(parent))
     #print(parents)
  
   #sys.stdout=original_stdout
  
       
  
  
  
  # Obtaining the parameters of node Tub
  xi = 0 #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  xi = 1  #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  xi = 2 #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  xi = 3  #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  xi = 4 #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  #xi = 5  #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  #xi = 6 #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  #xi = 7  #Tub is node 1
  factor = cppet.get_factor(num_vars + xi)
  parameters = factor.get_prob()
  #print(parameters)
  
  
  # ----INFERENCE---- #
  # Evidence propagation 
  # Set evidence. For example, Asia = 'yes' and Lung Cancer = 'no'
  cppet.set_evidence([0,3],[0,1])
  # Propagate evidence
  cppet.sum_compute_tree()
  # Get factor with results
  factor = cppet.get_factor(-1)
  prob = factor.get_prob() # The result is the probability of the evidence 
  # Retract evidence
  
  cppet.retract_evidence()
  
  # Obtaining most probable explanations (MPEs)
  # Set evidence. For example, Asia = 'yes' and Lung Cancer = 'no'
  cppet.set_evidence([0,3],[0,1])
  # Compute MPE
  cppet.max_compute_tree()
  # Get factor with results
  factor = cppet.get_factor(-1)
  mpe_idx = factor.get_mpe()[0] # Get the MPE 
  mpe = [var_classes[i][ci] for i,ci in enumerate(mpe_idx)]
  #print("Am printing")
  prob = factor.get_prob() # Get probability of the MPE 
  #print(prob)
  # Retract evidence
  cppet.retract_evidence()
