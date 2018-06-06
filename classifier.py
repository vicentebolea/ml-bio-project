import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

path = 'data_set/TCGA_6_Cancer_Type_Mutation_List.csv'
# Read column names from file
cols = list(pd.read_csv(path, nrows =1))
df = pd.read_csv(path)
df=df[df['Chromosome'] != 'X']
df=df[df['Chromosome'] != 'Y']

df= df[df['Chromosome'] !='GL000209.1']
df= df[df['Chromosome'] !='GL000212.1']

df= df[df['Chromosome'] !='GL000192.1']

df= df[df['Chromosome'] !='GL000205.1']
df= df[df['Chromosome'] !='GL000218.1']
df= df[df['Chromosome'] !='GL000213.1']


df= df[df['Chromosome']!="MT"]
X=df.drop('Cancer_Type',axis=1)
# X= X.drop('Gene_Name', axis=1)
# X=X.drop('Tumor_Sample_ID', axis=1)
# X=X.drop('Start_Position',axis=1)
# X = X.drop('End_Position', axis=1)
# X= X.drop('Variant_Type',axis=1)
encoder = LabelEncoder()

variant_Data = X.Variant_Type.values;
gene_name_data= X.Gene_Name.values;
tumor_sample_id= X.Tumor_Sample_ID.values;

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.
encoder.fit(variant_Data)
encoded_variant_type = encoder.transform(variant_Data)
X["Variant_Type"]=encoded_variant_type

encoder.fit(gene_name_data)
encoded_gene_name = encoder.transform(gene_name_data)
X['Gene_Name']=encoded_gene_name

encoder.fit(tumor_sample_id)
encoded_tumor_id = encoder.transform(tumor_sample_id)

X["Tumor_Sample_ID"]=encoded_tumor_id;
# X= X.drop('Reference_Allele', axis=1)
reference_allele= X.Reference_Allele.values;
encoder.fit(reference_allele)
encoded_reference_allele = encoder.transform(reference_allele)
X['Reference_Allele']=encoded_reference_allele

# X=X.drop('Tumor_Allele',axis=1)
tumor_Allele= X.Tumor_Allele.values;
encoder.fit(tumor_Allele)
encoded_tumor_Allele = encoder.transform(tumor_Allele)
X['Tumor_Allele']=encoded_tumor_Allele
# X= pd.DataFrame(df, usecols =[i for i in cols if i != 'Cancer_Type' and i != "Tumor_Sample_ID"])
y=df['Cancer_Type']
# y= pd.DataFrame(df, usecols =[i for i in cols if i == 'Cancer_Type'])
# X.fillna(value='X', inplace=True)
# X['Chromosome'].map({'X': 0})
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)



###############################################################################
# Train a SVM classification model




clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred_rf)
print(acc_score)
print(y_pred_rf)


