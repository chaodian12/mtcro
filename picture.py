import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
variables = []
with open('../SNP_pca/DTA.Y', 'r') as file:
    lines = file.readlines()
    variable_name = lines[0].strip()
    variable_values = [float(line.strip()) for line in lines[1:]]
    variables.append((variable_name, variable_values))
with open('../SNP_pca/KNPE.Y', 'r') as file:
    lines = file.readlines()
    variable_name = lines[0].strip()
    variable_values = [float(line.strip()) for line in lines[1:]]
    variables.append((variable_name, variable_values))
with open('../SNP_pca/KWPE.Y', 'r') as file:
    lines = file.readlines()
    variable_name = lines[0].strip()
    variable_values = [float(line.strip()) for line in lines[1:]]
    variables.append((variable_name, variable_values))
with open('../SNP_pca/PH.Y', 'r') as file:
    lines = file.readlines()
    variable_name = lines[0].strip()
    variable_values = [float(line.strip()) for line in lines[1:]]
    variables.append((variable_name, variable_values))

data = pd.DataFrame({name: values for name, values in variables})

correlation_matrix = data.corr().abs()

# 创建热力图，使用橙色调，并限制颜色范围为0到1
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, vmin=0, vmax=1)

plt.title('Maize 1404 Dataset Traits\' Pearson')

plt.show()
