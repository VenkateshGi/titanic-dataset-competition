import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
#list the downloaded csv file from kaggle in the current directory 
#, so that you can read the bothtrain.csv and test.csv files
# Any results you write to the current directory are saved as output.

titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")

titanic_train.info()

titanic_test.info()


# As we can see, there are null values in both train adn test datasets

#Let's set the survived to -1 and then we combine both datasets
titanic_test["Survived"] = -1
titanic = titanic_train.append(titanic_test)


# In[6]:


#let's check it now
titanic.Survived.value_counts()


# In[7]:


titanic.sample(3)


# So now, 
# Age,Fare are continuous variables and remaining are categorical. So let's plot boxplot for continuos variables and see whether there are any outliers

# **ANALYSIS FOR CONTINUOUS VARIABLES**

# In[8]:


titanic[["Age","Fare"]].boxplot()


# Everything is fine. These outliers can be understood. The Fare > 500 is outlier. They might bought the ticket lately.
# 
# **NOTE **: np.nan values are removed by defalut when we plot boxplot. I will fill those nan values in coming lines

# In[9]:


titanic[titanic.Fare > 500]


# Passengers whose fare is above 500, have survived the Titanic Disaster.

# **CATEGORICAL DATA ANALYSIS**

# In[10]:


temp = titanic_train.groupby("Sex")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="Sex",y = "percentage",hue = "Survived",data = temp).set_title("GENDER _ SURVIVAL")


# As we see, **80%** of **dead** passengers are **male** and around **75%** of **alive** passengers are **female**

# Let's understand Parch,SibsP more clearly

# In[11]:


temp = titanic[(titanic.Survived!=-1)].groupby("SibSp")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="SibSp",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")


# In[12]:


temp = titanic[(titanic.Survived!=-1)].groupby("Parch")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="Parch",y = "percentage",hue = "Survived",data = temp).set_title("Parch vs Survival Rate")


# From above plots we can understand that,
# 1. SibSp - (1,2) have good chances of survival
# 2. Parch - (1,2,3) have better chances of survival

# In[13]:


titanic.sample()


# Let's drop the columns that are not necessary 


titanic.drop(columns=["Ticket","PassengerId","Cabin"], inplace = True)


# Let's fill null values and refine our data



titanic.isna().sum()


# ***AGE***
# * So, we have to fill the Nan values of Age column. We can do this by 
# 1. mean/median/mode imputing where we can generalize the values and just fill them.
# 2. Either by using Interpolation. Here we can't interpolate data since there's no order to follow.
# 3. There may be some other methods to fill AGE attribute, like, Averaging the age based on Embark and Fare and Gender.
# 4. Here, I'm considering to fill the Nan Values of AGE using the **Title** of Name. For Example, Moran, **Mr**. James, here "Mr" is the title name and we can allocate him the Average of all passengers bearing this Title. 

# In[16]:


titanic[titanic.Age.isna()].head(3)


# In[17]:


"""
Storing the titles of passengers in title list and then adding it to titanic dataframe
"""
title = []
for item in titanic.Name:
    title.append(item.split(',')[1].split('.')[0].strip())
print (title[:3])
print (titanic.Name[:3])
titanic["title"] = title

titanic.title.value_counts()

using = dict(titanic.groupby("title").mean()["Age"])
sns.barplot(x = list(using.keys()), y = list(using.values()))
plt.xticks(rotation = 90)


# Different Passengers have different Age based on title. So, our assumption of filling Nan values is correct. Let's update Age accordingly

# In[20]:


final_age = []
for i in range(len(titanic)):
    age = titanic.iloc[i,0]
    if np.isnan(age):
        age = using[titanic.iloc[i,-1]]
    final_age.append(age)
titanic["Age"] = final_age


# In[21]:


titanic.isna().sum()


# ***Embarked***

# In[22]:


sns.countplot(x="Embarked", data = titanic)


# As you expected, filling with "S" for nan values is correct.

# In[23]:


titanic.Embarked.fillna("S",inplace=True)


# In[24]:


titanic.isna().sum()


# ***Fare***

# In[25]:


sns.barplot(x="Embarked",y="Fare",hue = "Pclass",data = titanic)


# In[26]:


titanic[titanic.Fare.isna()]


# So based on the above plot, 
# *  Thomas Embarked at S and Pclass ticket is of type 3, So let's assign Fare value as 18

# In[27]:


titanic.Fare.fillna(18,inplace=True)
titanic.isna().sum()


# So now, the data is clean. Let's see the analysis

# In[28]:


temp = titanic[(titanic.Survived!=-1)].groupby("Parch")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="Parch",y = "percentage",hue = "Survived",data = temp).set_title("Parch vs Survival Rate")


# Let's create is_par column, where if Parch!=0, is_par = 1, else is_par = 0

# In[29]:


Parch = titanic.Parch.tolist()
is_par = [0 if item == 0 else 1 for item in Parch ]
titanic["is_par"] = is_par


# In[30]:


temp = titanic[(titanic.Survived!=-1)].groupby("SibSp")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="SibSp",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")


# Let's apply the same for sibling also

# In[31]:


SibSp = titanic.SibSp.tolist()
has_sib = [0 if item == 0 else 1 for item in SibSp ]
titanic["has_sib"] = has_sib


# In[32]:


sns.countplot(x = "Embarked", hue = "Survived",data = titanic[titanic.Survived != -1])


# In[33]:


temp = titanic[(titanic.Survived!=-1)].groupby("Pclass")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")
sns.barplot(x="Pclass",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")


# Pclass 
# * 1 -> 60% have survived
# * 2 -> 45% have survived
# * 3 -> 23% have survived

# In[34]:


titanic.sample(4)


# In[35]:


titanic[titanic.Survived!=-1].groupby("Survived").mean()[["Age","Fare"]]


# So, we consider Age and Fare attributes also for modeling.

# For building the model, let's finalize the features.

# In[36]:


titanic.drop(columns=["Name","Parch","SibSp","title"], inplace=True)
titanic.sample()


# In[37]:


titanic = pd.get_dummies(titanic, columns=["Embarked","Pclass"])
titanic.Sex = titanic.Sex.map({"male":1,"female":0})
titanic.sample()


# So now, The data is ready for modeling

# In[38]:


titanic_training_y = titanic[titanic.Survived!=-1].Survived
titanic_training_x = titanic[titanic.Survived!=-1].drop(columns = ["Survived"])
from sklearn.model_selection import train_test_split
for random in range(15):
    train_x, test_x, train_y, test_y = train_test_split(titanic_training_x, titanic_training_y, test_size = 0.1)
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    scores = []
    for i in range(5,15):
        model = XGBClassifier(max_depth = i)
        model.fit(train_x, train_y)
        target = model.predict(test_x)
        score = accuracy_score(test_y, target)
        scores.append(score)
    print("best scores: ",max(scores), " at depth : ",scores.index(max(scores))+5)


# I would like to take depth as 6, and find the predictions

# In[39]:


titanic_training_y = titanic[titanic.Survived!=-1].Survived
titanic_training_x = titanic[titanic.Survived!=-1].drop(columns = ["Survived"])
test_x = titanic[titanic.Survived==-1].drop(columns = ["Survived"])
model = XGBClassifier(max_depth = i)
model.fit(titanic_training_x, titanic_training_y)
target = model.predict(test_x)
print (target[:4])
print (test_x[:4])

titanic_test = pd.read_csv("../input/test.csv")
titanic_test = pd.DataFrame(titanic_test["PassengerId"])
titanic_test["Survived"] = target
titanic_test.head()
titanic_test.to_csv("predictions.csv")

