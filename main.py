# Importando as bibliotecas utilizadas no projeto

# Database
import pandas as pandas

# Plotting
from matplotlib import pyplot as plotter
import seaborn as seaborn

# Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Utilities
import numpy as numpy

# Importando o dataset
breast_cancer_dataset = pandas.read_csv("breast_cancer_dataset.csv")

# Visualizando o dataset
print(breast_cancer_dataset.head())

# Limpando coluna Unnamed: 32

breast_cancer_dataset.drop("Unnamed: 32", axis = 1, inplace = True)
print(breast_cancer_dataset.isna().any())

# Fazendo plot de dados relevantes (fase exploratória)

scatter_x = breast_cancer_dataset['radius_mean']
scatter_y = breast_cancer_dataset['diagnosis']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Radius Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# --------

scatter_x = breast_cancer_dataset['area_mean']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Area Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# --------

scatter_x = breast_cancer_dataset['perimeter_mean']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Perimeter Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# --------

scatter_x = breast_cancer_dataset['texture_mean']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Texture Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# --------

scatter_x = breast_cancer_dataset['compactness_mean']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Compactness Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# --------

scatter_x = breast_cancer_dataset['concavity_mean']

plotter.scatter(scatter_x, scatter_y)
plotter.xlabel('Concavity Mean')
plotter.ylabel('Diagnosis')
plotter.show()

# Mais dados disponíveis no Google Colab

# Fazendo plot com Seaborn para visualizar possíveis outliers

seaborn.boxplot(x = 'radius_mean', data = breast_cancer_dataset)
seaborn.boxplot(x = 'area_mean', data = breast_cancer_dataset)
seaborn.boxplot(x = 'perimeter_mean', data = breast_cancer_dataset)

# Fazendo plot com Seaborn agora relacionando os dados de diagnóstico

seaborn.boxplot(x = 'area_mean', y = 'diagnosis', data = breast_cancer_dataset)
seaborn.boxplot(x = 'radius_mean', y = 'diagnosis', data = breast_cancer_dataset)
seaborn.boxplot(x = 'perimeter_mean', y = 'diagnosis', data = breast_cancer_dataset)
seaborn.boxplot(x = 'area_worst', y = 'diagnosis', data = breast_cancer_dataset)
seaborn.boxplot(x = 'radius_worst', y = 'diagnosis', data = breast_cancer_dataset)
seaborn.boxplot(x = 'perimeter_worst', y = 'diagnosis', data = breast_cancer_dataset)

# Fazendo um pré processamento da coluna de diagnóstico

encoder = LabelEncoder()
breast_cancer_dataset['diagnosis'] = encoder.fit_transform(breast_cancer_dataset['diagnosis'])

# Fazendo correlação entre os dados e fazendo um heatmap

correlation_matrix = breast_cancer_dataset[['diagnosis', 'area_mean', 'area_worst', 'perimeter_mean', 'perimeter_worst', 'radius_mean', 'radius_worst', 'smoothness_mean']].corr()
seaborn.heatmap(correlation_matrix, annot = True, linewidths = 0.1)
plotter.show()

# Definindo as variáveis de entrada e saida

cross_data = breast_cancer_dataset[['area_mean', 'perimeter_mean', 'radius_mean', 'area_worst', 'perimeter_worst', 'radius_worst']]
target_data = breast_cancer_dataset['diagnosis']

# Fazendo a separação entre teste e treinamento

cross_data_train, cross_data_test, target_data_train, target_data_test = train_test_split(cross_data, target_data, test_size = 0.2, stratify = target_data, random_state = 8)

# Descobrindo o melhor valor de K para o KNN

error = []

for i in range(1, 10):
  knn = KNeighborsClassifier(n_neighbors = i)
  knn.fit(cross_data_train, target_data_train)
  prediction_i = knn.predict(cross_data_test)
  error.append(numpy.mean(prediction_i != target_data_test))

plotter.figure(figsize = (12, 6))
plotter.plot(range(1, 10), error, color = 'red', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 10)

plotter.title('Error Rate K Value')
plotter.xlabel('K Value')
plotter.ylabel('Mean Error')

plotter.show()

# Preparando o pipeline

knn_pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors = 6))])

# Treinando o modelo

knn_pipeline.fit(cross_data_train, target_data_train)

# Fazendo a previsão e medindo a acuracia do modelo usando o accuracy_score

target_data_preddicted = knn_pipeline.predict(cross_data_test)
print(accuracy_score(target_data_test, target_data_preddicted))

# Fazendo um teste simples da primeira entrada

# Câncer benigno é 0 e maligno 1

if target_data_preddicted[0] == 0:
  print("\n O tumor provavelmente é benigno")
else:
  print("\n O tumor provavelmente é maligno")

# Treinando modelo LinearSVC e medido a acuracia do modelo usando o accuracy_score

svm_pipeline = Pipeline([("linear_svc", LinearSVC(C = 1))])
svm_pipeline.fit(cross_data_train, target_data_train)
svm_target_data_preddicted = svm_pipeline.predict(cross_data_test)

print(accuracy_score(target_data_test, svm_target_data_preddicted))

# Fazendo uma validação de feature importance

permutation = permutation_importance(knn_pipeline, cross_data_test, target_data_test)

feature_importance = {}

for i, feature in enumerate(cross_data.columns):
  feature_importance[feature] = permutation.importances_mean[i]

features_dataframe_pfi = pandas.DataFrame({
    "Feature": list(feature_importance.keys()),
    "Importance": list(feature_importance.values())
})

features_dataframe_pfi = features_dataframe_pfi.sort_values(by = "Importance", ascending = False)

print("\nImportância das Características (Permutation Feature Importance para KNN):")
print(features_dataframe_pfi)

# Plotando as informações

plotter.figure(figsize = (10, 6))
seaborn.barplot(x = 'Importance', y = 'Feature', data = features_dataframe_pfi, palette = 'viridis')
plotter.title('Permutation Feature Importance (KNN)')
plotter.xlabel('Importância (Queda na Acurácia)')
plotter.ylabel('Característica')
plotter.show()
