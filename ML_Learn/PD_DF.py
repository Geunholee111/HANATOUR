import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
pandas_DF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])

print(pandas_DF)
print(type(pandas_DF))

print(pandas_DF.head(10))
print(pandas_DF.shape)

print(pandas_DF.info())
print(pandas_DF.describe())

print(pandas_DF['sepal_width'].value_counts())

year_feature = pandas_DF['sepal_width']
print(year_feature.head(10))
year_value = pandas_DF['sepal_width'].value_counts()
print(year_value)

print(pandas_DF.columns)