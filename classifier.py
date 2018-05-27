import numpy as np
import pandas as pd

path = 'data_set/TCGA_6_Cancer_Type_Mutation_List.csv'
# Read column names from file
cols = list(pd.read_csv(path, nrows =1))
X= pd.read_csv(path, usecols =[i for i in cols if i != 'Cancer_Type'])
y= pd.read_csv(path, usecols =[i for i in cols if i == 'Cancer_Type'])


