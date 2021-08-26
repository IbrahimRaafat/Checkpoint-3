#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv("titanic-passengers.csv",sep=';')

def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 },

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }
    )


cat_gender={'Sex':{'male':0,'female':1}} #maping gender to binary
surv_to_bin={'Survived':{'No':0,'Yes':1}} # maping survival to binary
df.replace(cat_gender,inplace=True)
df.replace(surv_to_bin,inplace=True)

one_hot=pd.get_dummies(df['Embarked'])
df=df.drop('Embarked',axis=1)
df=df.join(one_hot)
df.sort_values(['Cabin'])   #sorting passengers acc to Ca No.
df=df.drop('Cabin',axis=1) #seems to be useless as there's too much missing data
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  #creating a new column(title) instead of name
df = df.drop(columns='Name')

df['Age'].fillna(int(df['Age'].mean()), inplace=True) #replace missing age values with column's mean


#minimizeing titles numbers to only 5 
df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4}) #same as we did with Sex &Embarked but in a different way



df["FamilySize"] = ""
df.FamilySize=df['SibSp']+df['Parch']
df=df.drop('Parch',axis=1)
df=df.drop('SibSp',axis=1)


#corr matrix
plot_correlation_map( df )


#Sex and survival
plt = df[['Sex', 'Survived']].groupby('Sex').mean().plot(kind='bar')
plt.set_xlabel('sex')
plt.set_ylabel('Survival Probability')

#Age and survival
plt2 = df[['Age', 'Survived']].groupby('Age').mean().plot(kind='area')
plt2.set_xlabel('Age')
plt2.set_ylabel('Survival Probability')

#Use the groupby function combined with the mean() to view the relation between Pclass and survived 

plt2 = df[['Pclass', 'Survived']].groupby('Pclass').mean().plot(kind='bar')
plt2.set_xlabel('Pclass')
plt2.set_ylabel('Survival Probability')

#Visualize the correlation between Title and other features(Survival Pclass)

plt3 = df[['Title', 'Survived']].groupby('Title').mean().Survived.plot(kind='bar')
plt3.set_xlabel('Title')
plt3.set_ylabel('Survival Probability')

plt4 = df[['Title', 'Pclass']].groupby('Title').mean().plot(kind='bar')
plt4.set_xlabel('Title')
plt4.set_ylabel('Pclass')




# In[ ]:


df.to_csv("modified.csv")

