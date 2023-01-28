import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier

def train_dt(x_train,y_train):
    """Train a decision tree classifier
    
    Args:
        x_train : training set
        y_train : training set labels
        
    Returns:
        dt : decision tree classifier"""
    dt = DecisionTreeClassifier(random_state=5) #Create a decision tree classifier
    dt.fit(x_train,y_train) #Fit the decision tree using the training set
    return dt

def score(name,X1,Health1):
    """permet de ca

    Args:
        name : name of the class to be studied

    Returns:
        F1 : Calculation of the F1 score of the model
        y_predict : Using the decision tree to predict the class membership of the test set instances
        accuracy : Calculation of the accuracy of the model
    """
    F1=[]
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X1,Health1, test_size=0.20, train_size=0.80,shuffle=True,stratify=Health1)
    dt = train_dt(x_train,y_train) #Fit the decision tree using the training set  
    y_predict = dt.predict(x_test) 
    F1.append(f1_score(y_test,y_predict,average="weighted")) 
    F1.append(f1_score(y_test,y_predict,average="macro"))
    accuracy=accuracy_score(y_test,y_predict)
    classificationreport=classification_report(y_test, y_predict)  
    return F1,accuracy,classificationreport,dt




def classification(name,df):
    """Classify the data into three categories: rich, normal and poor

    Args:
        name : label of an attribute
        df : dataframe

    Returns:
        df : dataframe
        X : dataframe without the label
        Health : list of the classification of the data
        y : list of the classification of the data
    """
    Health=[]
    compteur=[0,0,0]
    mean=np.mean(df[name]) #to know the average of the column Current health expenditure per capita (current US$)
    std=np.std(df[name]) #to know the standard deviation of the column Current health expenditure per capita (current US$)
    for i in df[name]:
        if i>mean+0.5*std:
            compteur[2]+=1
            Health.append("developed")
        elif i<mean-0.5*std:
            compteur[0]+=1
            Health.append("underdeveloped")
        else:
            compteur[1]+=1
            Health.append("emerging")
    df_modif=df.copy()
    y=df_modif[name]
    for i in range(0,len(y)):
        if y[i]>mean+0.5*std:
            y[i]="developed"
        elif y[i]<mean-0.5*std:
            y[i]="underdeveloped"
        else:
            y[i]="emerging"
    X=df_modif.drop(columns=[name],axis=1)
    return df_modif,X,Health,y
