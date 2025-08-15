import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r"C:\Users\konth\Downloads\Salary_Data - Salary_Data.csv")
#in the above datset we have 1 inddependent and 1 dependent  var
#so we have to use y=mx+c formula
#then salary=m(experience)+c
#now we have to sepereate dependent and independent columns
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
#there is no missiningvalues so we can skip impute part
#to perform test and strain use below line
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#we mentioned testsize as 0.20 so that defaulty train size will be 0.80
#random_state=0 means take fixed values/records
from sklearn.linear_model import LinearRegression
#select LinearRegresiion and press cltr+i to see the functionality and usage
regressor=LinearRegression()
# we are training the data by data using linearregression model
regressor.fit(x_train,y_train)
#predicting the x_test value
y_pred=regressor.predict(x_test)
#now compare actualdata,y_test,y_pred values by columns numbers
#to compare and print actual data and predicted data
comparison=pd.DataFrame({'actual':y_test,'predicted':y_pred})
print(comparison)
#visulazation using graph
#to see and check best line
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of expereince')
plt.ylabel('salary')
plt.show()
#now open comparison var and check the graph
#to see slope value
m_slope=regressor.coef_
print(m_slope)
#to see intercept value
c_intercept=regressor.intercept_
print(c_intercept)
#to calculate zscore import module
import scipy.stats as stats
#zscore for entire dataset
print(dataset.apply(stats.zscore))
#zscore for separate column
print(stats.zscore(dataset['Salary']))
#for no.of rows
a=dataset.shape[0]
#for no.of columns
b=dataset.shape[1]
#to find the degrees of freedom
degree_of_freedom=a-b
print(degree_of_freedom)
#to calculate the sume of squares regressio(SSR)
#calculate mean of dependentt variable
y_mean=np.mean(y)
#SSR
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)
#sum of suares erro(SSE)
y=y[0:6]#beacuase we used only 6 records in y_test
SSE=np.sum((y-y_pred)**2)
print(SSE)
#sume sqares total(SST)
SST=SSR+SSE
print(SST)
#to calculate r square
r_square=1-(SSR/SST)
print(r_square)#we build a good model because of we re getting r-sq value bw 0to1
#to calculate bias/train score
bias=regressor.score(x_train,y_train)
print(bias)
#to calculate varience/test score
variance=regressor.score(x_test,y_test)
print(variance)
#mean_square_error
from sklearn.metrics import mean_squared_error
train_mse=mean_squared_error(y_train,regressor.predict(x_train))
test_mse=mean_squared_error(y_test,y_pred)
#to decrease the size and every ML model stored in the form of pickle
#when size is less we can easily integreate with frontend
import pickle
#save the trained model to disk
filename='linear_regression_model.pkl'
#open file in write-binary mode and dump the model
with open(filename,'wb')as file:
    pickle.dump(regressor,file)
print("model has been pickled and saved as linear_regressio_model")
#to the see the location of pickled file
import os
os.getcwd()
