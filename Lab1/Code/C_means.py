# implemented by python sklearn.cluster
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MyK_means
from MyK_means import txt2csv, drawCategoryFigure
from Settings import *

sns.set(style="darkgrid")

import warnings

warnings.filterwarnings("ignore")

datasetIndex = input(
    "Please enter a dataset index number: (1)iris.csv (2)sonar.csv (3)Compound.txt (4)threecircles.txt: ")
x = []
dataset = []
if datasetIndex == '1':
    df_iris = pd.read_csv('iris.csv')
    print("info...")
    df_iris.info()
    df_iris = df_iris.drop(['Unnamed: 0'], axis=1)  # drop the first column
    # df_iris.describe()
    print(df_iris.head())
    sns.pairplot(df_iris, hue="Species")  # https://zhuanlan.zhihu.com/p/98729226
    plt.show()
    x = df_iris.iloc[:, [0, 1, 2, 3]].values  # Get the first four column's value(remember we delete the first column)
elif datasetIndex == '2':
    df_sonar = pd.read_csv('sonar.csv', header=None)
    df_sonar.info()
    print(df_sonar.head())
    x = df_sonar.iloc[:, 0:60].values
elif datasetIndex == '3':
    # change txt format to csv format
    txt2csv('Compound.txt', 'Compound.csv')
    df_compound = pd.read_csv('Compound.csv', header=None, names=['a', 'b', 'c'])
    df_compound.info()
    print(df_compound.head())
    sns.pairplot(df_compound, hue="c")
    plt.show()
    x = df_compound.iloc[:, 0:2].values
else:
    txt2csv('threecircles.txt', 'threecircles.csv')
    df_threecircles = pd.read_csv('threecircles.csv', header=None, names=['a', 'b', 'c'])
    df_threecircles.info()
    print(df_threecircles.head())
    sns.pairplot(df_threecircles, hue="c")
    plt.show()
    x = df_threecircles.iloc[:, 0:2].values

print("===========================================================================")
kmeansKind = input("Which subLab do you want to test? (1)K-means (2)K-means++ (3)K-means with kernel function: ")
kernelKind = ''
if kmeansKind == '1':
    kmeansKind = 'kmeans'
elif kmeansKind == '2':
    kmeansKind = 'kmeans++'
else:
    kmeansKind = 'kmeans_kernel'
    kernelKind = input("Which kind of kernel would you like? (1)gauss (2)linear: ")
    if kernelKind == '1':
        MyK_means.global_kernel_kind = 'gauss_kernel'
    elif kernelKind == '2':
        MyK_means.global_kernel_kind = 'linear_kernel'

# for classification
from sklearn.cluster import KMeans

wcss = []
result = []
results = []
returnAllcenters = []

# 1.use python library
if use_python_library:
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        # evaluate css, just like the homework needs
        wcss.append(kmeans.inertia_)  # https://www.jianshu.com/p/cd7f3908a418
        print("insystem...", kmeans.inertia_)
else:
    for i in range(1, 11):
        kmeans = MyK_means.MyKMeans(i, 300, 10, kmeansKind)
        dataset, result, returnCenters = kmeans.kMeans(x)
        results.append(result)
        returnAllcenters.append(returnCenters)
        print("inmain...", kmeans.inertia_)
        wcss.append(kmeans.inertia_)

# plotting
plt.plot(range(1, 11), wcss)
plt.title('The elbow method of dataset ' + datasetIndex)
plt.xlabel('Numbers of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()

firstTime = True
if datasetIndex != '2':
    showRes = 'Y'
    while showRes == 'Y' or showRes == 'y':
        print("===========================================================================")
        if firstTime:
            showRes = input("do you want to check the category results? Y/N: ")
            firstTime = False
        else:
            showRes = input("do you want check another category results? Y/N: ")
        if not (showRes == 'y' or showRes == 'Y'):
            break
        clusterNum = input("how many clusters will you choose?: ")
        if datasetIndex == '1':
            df_iris['result'] = results[int(clusterNum) - 1]
            sns.pairplot(df_iris, hue="result")
            plt.show()
        elif datasetIndex == '3':
            df_compound['result'] = results[int(clusterNum) - 1]
            sns.pairplot(df_compound, hue="result")
            plt.show()
        else:
            df_threecircles['result'] = results[int(clusterNum) - 1]
            sns.pairplot(df_threecircles, hue="result")
            plt.show()

        if datasetIndex == '3' or datasetIndex == '4':
            df_data = []
            if datasetIndex == '3':
                df_data = df_compound
            else:
                df_data = df_threecircles

            if kmeansKind == 'kmeans_kernel':
                drawCategoryFigure(results[int(clusterNum) - 1], dataset, returnAllcenters[int(clusterNum) - 1], 3, df_data)
            else:
                drawCategoryFigure(results[int(clusterNum) - 1], dataset, returnAllcenters[int(clusterNum) - 1], 2, df_data)


