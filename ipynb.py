
# coding: utf-8

#    
#  ### NAME: Marmik S Modi
#  
# # TASK 1 : PREDICTION USING SUPERVISED ML
# 
# 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# READING THE DATASET

# In[43]:


df = pd.read_csv('http://bit.ly/w-data')
df


# In[3]:


df.isnull==True  #To check wheter it contains null value or not


#  As there are no null values we can now visualize the input data

# In[4]:


df.head() #DISPLAY TOP FIVE VALUE


# In[5]:


df.shape   #Shape of the data frame


# In[6]:


df.plot(x="Hours",y="Scores",style='o')
plt.title("Hours vs Percentage")
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored')
plt.show()


# #### From the visualization we can observe that there is a correlation between No. of Hours studied and Percentage Scored .So, plotting the regression line to confirm the correlation.

# In[7]:


sns.regplot(x=df['Hours'],y=df['Scores'])
plt.title("Regression Plot",size =20)
plt.xlabel('Hours Studied',size=15)
plt.ylabel('Percentage Scored',size=15)
plt.show()
print(df.corr())


# #### Variables are positively correlated

# In[8]:


#defining X and y from the data  
X= df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


#Spliting the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[18]:


#Fitting the Model
linear_regression=LinearRegression()
linear_regression.fit(X_train,y_train)
print('Model Trained Successfully')


# ### Predicting the Percentage of Marks

# In[17]:


pred_y=linear_regression.predict(X_test)
prediction=pd.DataFrame({'Hours':[i[0] for i in X_test],'Predicted Score':[k for k in pred_y]})
prediction


# In[20]:


#Comparing the values
compare_marks = pd.DataFrame({'Actual Marks':y_test,'Predicted Marks':pred_y})
compare_marks
    


# In[25]:


#VISUALIZING THE ACTUAL AND PREDICTED MARKS

plt.scatter(x=X_test, y=y_test, color='cyan')
plt.plot(X_test, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[27]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(y_test,pred_y))


# Small mean absolute error shows that there is very less chances of error through this model

# ## What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[42]:


hours =[9.25]
ans = linear_regression.predict([hours])
print("Scores =",ans[0].round(2))


# ## Hence according to the regression model if the Student studies for 9.25hours/day he/she would likely to get 93.89 Marks
