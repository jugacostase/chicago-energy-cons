#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CME594 Introduction to Data Science
Homework 7 Code - Decision Tree Learning
@author: Sybil Derrible
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')



#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'course_data'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)


#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X and Y
X = data.drop(columns=['Letter', 'Grade'])
Y = data.Grade  #Grade will be our target varibale (Hevar)


#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Scaling the data using StandardScaler
scaler = preprocessing.StandardScaler().fit(X_train)
#Scaling the data using MinMaxScaler
#scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train_scaled = X_train #scaler.transform(X_train) NO SCALING THIS WEEK TO LOOK AT SHAP VALUES
X_test_scaled = X_test #scaler.transform(X_test)


#Fit the Decision tree
cross_val_number = 5
estimators = 50 #This is number of our trees (Hevar)
decitree = tree.DecisionTreeRegressor() #using default criterion
rf = RandomForestRegressor(n_estimators=estimators, max_samples=30)
gbm = GradientBoostingRegressor(n_estimators=estimators, learning_rate=0.1, max_depth=2) #learning_rate will helps for overfitting


#Cross Validation (CV) process
decitree_scores = cross_val_score(decitree, X_train_scaled, Y_train, cv=cross_val_number)
rf_scores = cross_val_score(rf, X_train_scaled, Y_train, cv=cross_val_number)
gbm_scores = cross_val_score(gbm, X_train_scaled, Y_train, cv=cross_val_number)
print("")
print("Decision Tree r2: {0} (+/- {1})".format(decitree_scores.mean().round(2), (decitree_scores.std() * 2).round(2)))
print("Random Forest r2: {0} (+/- {1})".format(rf_scores.mean().round(2), (rf_scores.std() * 2).round(2)))
print("Gradient Boosting r2: {0} (+/- {1})".format(gbm_scores.mean().round(2), (gbm_scores.std() * 2).round(2)))
print("")


#Training final algorithms
decitree.fit(X_train_scaled, Y_train) #Decision Tree fitting
rf.fit(X_train_scaled, Y_train) #Random Forest fitting
gbm.fit(X_train_scaled, Y_train) #Gradient Boosting fitting


#Final Predictions
print("Y test values: {0}".format(Y_test.values))
print("")

#Decision treee
decitree_predict = decitree.predict(X_test_scaled)
decitree_score = metrics.r2_score(decitree_predict, Y_test)
print("Decision Tree: {0}".format(decitree_predict.round(2)))
print("Decision tree score: {0}".format(decitree_score.round(2)))
print("")

#Random Forest
rf_predict = rf.predict(X_test_scaled)
rf_score = metrics.r2_score(rf_predict, Y_test)
print("Random Forest: {0}".format(rf_predict.round(2)))
print("Random Forest score: {0}".format(rf_score.round(2)))
print("")

#Gradient Boosting
gbm_predict = gbm.predict(X_test_scaled)
gbm_score = metrics.r2_score(gbm_predict, Y_test)
print("Gradient Boosting: {0}".format(gbm_predict.round(2)))
print("Gradient Boosting score: {0}".format(gbm_score.round(2)))
print("")


#Feature Importance
#Making a dataframe of the results
decitree_FI = pd.DataFrame(decitree.feature_importances_, index=X.columns, columns=["FI"])
rf_FI = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["FI"])
gbm_FI = pd.DataFrame(gbm.feature_importances_, index=X.columns, columns=["FI"])
#Making the figure
fig = plt.figure() #Define an empty figure
ax0 = fig.add_subplot(131)
plt.title("Decision Tree")
sns.barplot(data=decitree_FI, x='FI', y=decitree_FI.index)
ax0 = fig.add_subplot(132)
plt.title("Random Forest")
sns.barplot(data=rf_FI, x='FI', y=rf_FI.index)
ax0 = fig.add_subplot(133)
plt.title("Gradient Boosting")
sns.barplot(data=gbm_FI, x='FI', y=gbm_FI.index)
fig.set_size_inches(18,6)
plt.tight_layout()
plt.savefig(file_name + '_FI.png') #Saving the plot
plt.show()


'''
SHAP
Useful resources include: 
# https://towardsdatascience.com/explaining-scikit-learn-models-with-shap-61daff21b12a
# https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html#Cohort-bar-plot
'''

explainer = shap.Explainer(gbm)
shap_values = explainer(X) #Entire dataset
shap_train = explainer(X_train) #Train set
shap_test = explainer(X_test) #Test set - You can count the 14 dots per feature (i.e., size of test set)

#Global Bar Plot
fig = plt.figure() #Define an empty figure
ax0 = fig.add_subplot(131)
shap.plots.bar(shap_values, show=False)
plt.title("Entire Dataset")
ax0 = fig.add_subplot(132)
shap.plots.bar(shap_train, show=False)
plt.title("Train Set")
ax0 = fig.add_subplot(133)
shap.plots.bar(shap_test, show=False)
plt.title("Test Set")
fig.set_size_inches(18,6)
plt.tight_layout()
plt.savefig(file_name + '_SHAP_bar.png') #Saving the plot
plt.show()


#Global Summary Plot
shap.summary_plot(shap_test, show=False)
plt.savefig(file_name + '_SHAP_summary.png') #Saving the plot
plt.show()


#Global Heatmap
shap.plots.heatmap(shap_test, cmap=plt.get_cmap("winter_r"), show=False) #show=False to save the plot otherwise it will show it but will not save it (Hevar)
plt.savefig(file_name + '_SHAP_heatmap.png') #Saving the plot
plt.show()


#Local Bar Plot
i = 1
print("Predicted: {0}".format(gbm_predict[i].round(2)))
print("Base value: {0}".format(shap_test[i].base_values))
print("Shap values: {0}:".format(shap_test[i].values))
base = round(shap_test[i].base_values[0],2)
sum_shap = round(shap_test[i].values.sum(), 2)
base_plus_shap = base + sum_shap
print("{0} + {1} = {2}".format(base, sum_shap, base_plus_shap))
shap.plots.bar(shap_test[i], show=False) #See that the sum of the SHAP values gives you the difference between the base value and the predicted value
plt.savefig(file_name + '_SHAP_local_barplot.png') #Saving the plot
plt.show()

'''
#Cohort Bar Plots
shap.plots.bar(shap_values.cohorts(data.Letter.values).abs.mean(0), show=False)
plt.savefig(file_name + '_SHAP_cohort_barplot.png') #Saving the plot
plt.tight_layout()
plt.show()
'''