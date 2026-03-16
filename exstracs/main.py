import pandas as pd
from skExSTraCS import ExSTraCS

#Read from CSV file
data = pd.read_csv("Multiplexer6.csv")

#Specify the dataset's phenotype label
classLabel = "class"

#Derive the attribute and phenotype array using the phenotype label
dataFeatures = data.drop(classLabel,axis = 1).values
dataPhenotypes = data[classLabel].values

#Optional: Retrieve the headers for each attribute as a length n array
dataHeaders = data.drop(classLabel,axis=1).columns.values

print("Data Features")
print(dataFeatures)
print("\nData Phenotypes")
print(dataPhenotypes)
print("\nData Headers")
print(dataHeaders)

model = ExSTraCS(learning_iterations = 10,rule_compaction=None,nu=10)

trainedModel = model.fit(dataFeatures,dataPhenotypes)

trainedModel.export_final_rule_population(dataHeaders,classLabel,filename="fileRulePopulation.csv",DCAL=False)




