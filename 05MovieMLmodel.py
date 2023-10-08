from sklearn.tree import DecisionTreeClassifier
import pandas

df=pandas.read_csv('moviechoice.csv')
# print(df)

features=df.drop(columns=['genre'])
# print(features)

labels=df['genre']

model=DecisionTreeClassifier()
model.fit(features.values,labels.values)

result=model.predict([[1,25,1],[1,43,2]])
print(result)

