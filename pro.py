from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
df=pd.read_csv(r'G:\dataset and python\diabetes.csv')
print(df)
feature_names=['Preganencies','glucose','bloodpressure','Skin Thickness','insulin','BMI','DiabetesPedigreeFunction','age']
X=df[feature_names]
y=df['Outcome']
train_input=[]
train_output=[]
for i in range(df.shape[0]):
   
    a=df['Preganencies'][i]
    b=df['glucose'][i]
    c=df['bloodpressure'][i]
    d=df['Skin Thickness'][i]
    e=df['insulin'][i]
    f=df['BMI'][i]
    g=df['DiabetesPedigreeFunction'][i]
    h=df['age'][i]
    train_input.append([a,b,c,d,e,f,g,h])
    i=df['Outcome'][i]
    train_output.append(i)
P=[[10,115,74,0,0,25.6,0.201,29]]

#{Decision Tree Model}
clf=DecisionTreeClassifier()
clf=clf.fit(X=train_input,y=train_output)
print("\n1)Using Decision Tree Prediction is "+str(clf.predict(P)))

#{K Neighbors Classifier}
Knn=KNeighborsClassifier()
Knn.fit(X=train_input,y=train_output)
print("2)Using K Neighbors Classifier Prediction is "+str(Knn.predict(P)))

#{Using RandomForest Classifier}
rfor=RandomForestClassifier()
rfor.fit(X=train_input,y=train_output)
print("3)Using RandomForestClassifier Prediction is "+str(rfor.predict(P))+"\n")

#Performance of the classifier
X=train_input
y=train_output

#Splitting the dataset into Training set and Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=9,random_state=0)


#Summary of the predictions made by the classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier=Knn
classifier.fit(X=X_train,y=y_train)
print("Using Chosen Classifier the Prediction is "+str(classifier.predict(P))+"\n")

y_pred=classifier.predict(X_test)

#Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
print("#===============Summary of the predictions made by the classifier")
print("No of test cases",len(y_test))
print()
for i in range(len(y_test)):
    if(y_test[i]!=y_pred[i]):
        print("\n Test Case:",y_test[i],"Prediction:",y_pred[i])
print("#=========confusion matrix")
label_lst=[1,0]
print("Label List for confusion martix:")
print(label_lst)
print(confusion_matrix(y_test,y_pred,labels=label_lst))
print("\nclassification_report:",classification_report(y_test,y_pred))
input("press enter")
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import scikitplot
from matplotlib import pyplot as plt
print(classification_report(y_test,y_pred))
scikitplot.metrics.plot_confusion_matrix(y_test,y_pred)
plt.show()

#Visualisations
from matplotlib import pyplot as plt

#more info on the data
print(df.info())
print(df['Preganencies'].unique());print(df['glucose'].unique());
print(df['bloodpressure'].unique());print(df['Skin Thickness'].unique());
print(df['insulin'].unique());print(df['BMI'].unique());
print(df['DiabetesPedigreeFunction'].unique());print(df['age'].unique())

#histograms
df.hist(edgecolor='black',linewidth=1.2)
plt.suptitle('HISTOGRAM')

#box and whisker plots
df.plot(kind='box',sharex=False,sharey=False,title='Box Plot')

#boxplot on each feature split out by Outcome
df.boxplot(by="Outcome",figsize=(10,10))
plt.suptitle('Box Plot by Outcome')

#scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

#scatter plot matrix for predictions vs actual values
X_test1=range(len(y_test))
plt.scatter(X_test1,y_test,color='blue',marker='^')
plt.plot(X_test1,y_pred,'ro')
plt.show()





      
              


    


