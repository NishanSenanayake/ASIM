# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:33:54 2020

@author: Nishan Senanayake
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



#Reading the CSV

data = pd.read_csv("My_csv.csv", nrows= 12) 

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------


#Removing features with low variance of processing numerical data 

processing_data= data.iloc[:, 6:135]

#normalizing the processing variables
# min_max_scaler = preprocessing.MinMaxScaler()
# processing_data_norm = min_max_scaler.fit_transform(processing_data)
# processing_data_norm = pd.DataFrame(processing_data_norm)
# processing_data_norm.columns = processing_data.columns


#Removing features with low variance of processing numerical data

sel = VarianceThreshold(threshold=0.0001)
sel.fit_transform(processing_data/processing_data.mean())

constant_columns= [column  for column in processing_data.columns
                    if column not in processing_data.columns[sel.get_support()]]




#------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# PSD feature selction using Pearson Correlation


process_data_no_cons= processing_data.drop(constant_columns, axis=1)


psd= process_data_no_cons.iloc[:, 0:69]


# plot  Pearson Correlation

plt.figure(figsize=(20,14))
cor = psd.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("psd_corr.png", dpi=100)
plt.show()

# select highly correlated features and remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > threshold) or (abs(corr_matrix.iloc[i, j]) < (-threshold)): # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr



#Selecting corelated columns
corr_psd_features = correlation(psd, 0.90)
len(set(corr_psd_features))
print(corr_psd_features)

# Dropping corelated columns

psd_non_cor= psd.drop(corr_psd_features,axis=1)
list(psd_non_cor.columns)


#plotting non-corlealted heatmap
plt.figure(figsize=(20,14))
non_cor = psd_non_cor.corr()
sns.heatmap(non_cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("psd_non_corr.png", dpi=100)
plt.show()



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


#Rhelogoy data check Correlation

rhelogy =process_data_no_cons.iloc[:, 69:86]



plt.figure(figsize=(14,14))
cor = rhelogy.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.savefig("rehelogy.png", dpi=300)
plt.show()

corr_rhelogy_features = correlation(rhelogy, 0.9)
len(set(corr_rhelogy_features))

# no rhelogy corelation 


#--------------------------------------------------------------------------------
#chemistry corlation
#--------------------------------------------------------------------------------

chemistry =process_data_no_cons.iloc[:, 86:129]




plt.figure(figsize=(20,14))
cor = chemistry.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("chemistry.png", dpi=100)
plt.show()




#selecting corelated chemistry columns
corr_chemistry_features = correlation(chemistry, 0.95)
len(set(corr_chemistry_features))
print(corr_chemistry_features)


# Dropping corelated columns

chemistry_non_cor= chemistry.drop(corr_chemistry_features,axis=1)
list(chemistry_non_cor.columns)


#plotting non-corlealted heatmap
plt.figure(figsize=(20,14))
non_cor = chemistry_non_cor.corr()
sns.heatmap(non_cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("chemistry_non_corr.png", dpi=100)
plt.show()











#--------------------------------------------------------------------------------------------------------
# Removing features with low variance
#-----------------------------------------------------------------------------------------------------


process_data = data.iloc[:, 6:135]
process_data = process_data .drop(corr_psd_features,axis=1)
process_data = process_data .drop(corr_chemistry_features,axis=1)
process_data = process_data .drop(constant_columns, axis=1)


X=process_data

#-------------------------------------------------------------------------------------------------
#Feature selction with Univariate Feature Selection
#-------------------------------------------------------------------------------------------------

# powder impact to the microstructure using Univariate feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression



featureSelector = SelectKBest(score_func=f_regression, k=30)
featureSelector.fit(X,y)
for i in range(len(featureSelector.scores_)):
	print('Feature %d: %f' % (i, featureSelector.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(featureSelector.scores_))], featureSelector.scores_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig("processing_porosity_vf.png", dpi=300)
plt.show()

features= pd.DataFrame(featureSelector.scores_,columns=["Score"])
feature_cols= pd.DataFrame(X.columns)
feature_im_prosity_vf= pd.concat([feature_cols,features], axis=1)


#-------------------------------------------------------------------------------------------------
#Feature selction with Forward selection
#-------------------------------------------------------------------------------------------------



# Sequential Forward Selection(sfs)
sfs = SFS(LinearRegression(),
          k_features=20,
          forward=True,
          floating=False,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = 4)
sfs.fit(X, y)
sfs.k_feature_names_


fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.ylabel('Performance')
plt.grid()
plt.savefig("forward_processing_linear_porosity_vf.png", dpi=300)
plt.show()






#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#Forward selection with ridge
#------------------------------------------------------------------------------------

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm 

def cal_residual(y_pred, y_org):
     return  (y_pred-y_org)


#Assign depent varibale
y1=data['Porosity_VF_GS'] 
y2=data['Porosity_Size_GS']
y3=data['Grain_Size_FHT']
y4= "crystal Struc"
y5= data['Avg_Carbide_Dia']
y6=data['Carbide_VF']
y7=data['Avg._Nitride_Dia']
y8=data['Nitride_VF']
y9= data["Grain_Structure_FHT"].astype('category')
y9=y9.cat.codes


#Assign depent varibale
name="y8"
y=y8
X_norm= ((X-X.min())/(X.max()-X.min()))

cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
sfs_ridge_forward= SFS(Ridge(alpha=0.01),
          k_features=5,
          forward=True,
          floating=True,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = cv)
sfs_ridge_forward.fit(X_norm, y)
sfs_ridge_forward.k_feature_names_


fig1 = plot_sfs(sfs_ridge_forward.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.ylabel('Perfomance')
plt.grid()
plt.savefig("forward_processing_ridge_"+name+".png", dpi=300)
plt.show()




#--------------------------------------------------------------------------------
#Rige using slected features from forward feature selection
#----------------------------------------------------------------------------------


X_selcted_columns= list(sfs_ridge_forward.k_feature_names_)
X_selected=X_norm[X_selcted_columns]
ridge=Ridge()
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=cv)
result_ridge= ridge_regressor.fit(X_selected,y)


print('MAE: %.3f' % result_ridge.best_score_)
print('Config: %s' % result_ridge.best_params_)


model=Ridge(alpha=0.1)
model.fit(X_selected,y)
prediction_ridge=model.predict(X_selected)
y_pred=prediction_ridge
# only for Grain_Structure
#prediction_ridge= np.round(prediction_ridge)


plt.figure(figsize=(6,6))
plt.plot(y,prediction_ridge, 'ro', alpha=0.5)
plt.ylabel('Predicted Value')
plt.xlabel('Actual Value')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
fmin = min(xmin, ymin)
fmax = max(xmax, ymax)
plt.xlim(fmin, fmax)
plt.ylim(fmin, fmax)
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'b')
plt.savefig(name+".png", dpi=300, bbox_inches = 'tight')
plt.show()
coefficient_of_dermination = r2_score(y, prediction_ridge)

print(model.intercept_)
columns_importance=  np.array(list(zip(sfs_ridge_forward.k_feature_names_, model.coef_)))


#----------------------------------------------------------------------------------------
#Check Assupmtions
#----------------------------------------------------------------------------------------

# 1 Independence  no correlation between consecutive residuals



residual = cal_residual(prediction_ridge,y)
print(durbin_watson(residual))
plt.plot(prediction_ridge,residual, 'ro', color='green', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('Residuals')
plt.xlabel('Predicted Value')
plt.savefig("ridge_residual"+name+".png", dpi=300, bbox_inches = 'tight')
plt.show()



sm.qqplot(residual, alpha=0.5) 
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'r')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.savefig("qq"+name+".png", dpi=300,bbox_inches = 'tight')
plt.show()





#------------------------------------------------------------------------------
#Structure to property
#------------------------------------------------------------------------------

structure_data = pd.read_csv("My_csv.csv", nrows= 18) 
structure=structure_data.iloc[:, 135:145]
structure['Grain_Structure_FHT']=structure["Grain_Structure_FHT"].astype('category')
structure['Grain_Structure_FHT']=structure['Grain_Structure_FHT'].cat.codes
X=structure
X_norm= ((X-X.min())/(X.max()-X.min()))


y1=structure_data['Hardness_FHT'] 
y2=structure_data['RaT']
y3=structure_data['RaL']
y4=structure_data['AvgHCFlife']
y5=structure_data['AvgHCFstress']
y6=structure_data['Elastic_Modulus']
y7=structure_data['Prop_Limit']
y8=structure_data['0.02_YS']
y9=structure_data['0.2_YS']
y10=structure_data['UTS']
y11=structure_data['Ef']
#only for NaN Values
y=y10
idx= np.where(pd.isnull(y))
y=y.drop(idx[0][0])
X_norm= X_norm.drop(idx[0][0])

cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
sfs_ridge_forward= SFS(Ridge(alpha=0.01),
          k_features=4,
          forward=True,
          floating=True,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = cv)
sfs_ridge_forward.fit(X_norm, y)
sfs_ridge_forward.k_feature_names_


fig1 = plot_sfs(sfs_ridge_forward.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.ylabel('Perfomance')
plt.grid()
plt.savefig("forward_processing_ridge_"+name+".png", dpi=300)
plt.show()



X_selcted_columns= list(sfs_ridge_forward.k_feature_names_)
X_selected=X_norm[X_selcted_columns]
ridge=Ridge()
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=cv)
result_ridge= ridge_regressor.fit(X_selected,y)


print('MAE: %.3f' % result_ridge.best_score_)
print('Config: %s' % result_ridge.best_params_)


model=Ridge(alpha=0.01)
model.fit(X_selected,y)
prediction_ridge=model.predict(X_selected)

# only for Grain_Structure
#prediction_ridge= np.round(prediction_ridge)


fig, ax1 = plt.subplots()
ax1.plot(y,prediction_ridge, 'ro')
ax1.set_ylabel('Predicted Value')
ax1.set_xlabel('Actual Value')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

fmin = min(xmin, ymin)
fmax = max(xmax, ymax)

plt.xlim(fmin, fmax)
plt.ylim(fmin, fmax)
plt.show()
coefficient_of_dermination = r2_score(y, prediction_ridge)


model.intercept_
columns_importance=  np.array(list(zip(sfs_ridge_forward.k_feature_names_, model.coef_)))




#-------------------------------------------------------------------------------------------------
#Feature selction with backward selection
#-------------------------------------------------------------------------------------------------

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
# Sequential Forward Selection(sfs)
sfs = SFS(LinearRegression(),
          k_features=20,
          forward=False,
          floating=False,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = 5)
sfs.fit(X, y)
sfs.k_feature_names_


fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Backward Selection (w. StdErr)')
plt.ylabel('Performance')
plt.xticks(rotation=90)
plt.grid()
plt.savefig("backward_processing_linear_porosity_vf.png", dpi=300)
plt.show()



#-------------------------------------------------------------------------------------------------
#Feature selction with backward selection with Ridge
#-------------------------------------------------------------------------------------------------

sfs_back_ridge = SFS(Ridge(alpha=100),
          k_features=20,
          forward=False,
          floating=False,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = 5)
sfs_back_ridge.fit(X, y)
sfs_back_ridge.k_feature_names_


fig1 = plot_sfs(sfs_back_ridge.get_metric_dict(), kind='std_err')
plt.title('Sequential Backward Selection (w. StdErr)')
plt.ylabel('Performance')
plt.xticks(rotation=90)
plt.grid()
plt.savefig("backward_processing_ridge_porosity_vf.png", dpi=300)
plt.show()











#-------------------------------------------------------------------------------------------------
#Feature selction with Ensamble Tree
#-------------------------------------------------------------------------------------------------
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X_norm,y)
ranked_features= pd.Series(model.feature_importances_, index= X.columns)
ranked_features.nlargest(30).plot(kind="barh")
feature_imp_extra_tree=pd.DataFrame(ranked_features.nlargest(20))
plt.figure(figsize=(14,14))

plt.savefig("extra_tree.png", dpi=300)
plt.show()

#-------------------------------------------------------------------------------------------------
#Feature selction with Random forest Tree
#-------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
model= RandomForestRegressor()



cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_norm, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
model.fit(X_norm,y)
ranked_features= pd.Series(model.feature_importances_, index= X_norm.columns)
ranked_features.nlargest(5).plot(kind="barh")
feature_imp_rf_tree=pd.DataFrame(ranked_features.nlargest(4))
plt.figure(figsize=(14,14))

X_selected= X_norm[pd.DataFrame(ranked_features.nlargest(4)).index]
model.fit(X_selected,y)
prediction_RF=model.predict(X_selected)




fig, ax1 = plt.subplots()
ax1.plot(y,prediction_RF, 'ro')
ax1.set_ylabel('Predicted Value')
ax1.set_xlabel('Actual Value')
# plt.setp(ax1.get_xticklabels(), ha="right", rotation=45)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

fmin = min(xmin, ymin)
fmax = max(xmax, ymax)

plt.xlim(fmin, fmax)
plt.ylim(fmin, fmax)
plt.show()
coefficient_of_dermination = r2_score(y, prediction_RF)

print(coefficient_of_dermination)

plt.savefig("RF_tree.png", dpi=300)
plt.show()













#---------------------------------------------------------------------------------
#PCA
#--------------------------------------------------------------------------------



#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# processing_data_pca= data.iloc[:, 6:135]
# X=processing_data_pca
# y= data.iloc[:,135]

scaler = StandardScaler()
scaler.fit(X)  
scaled_X = scaler.transform(X)
pca = PCA()
pca.fit(scaled_X)
pca_data=pca.transform(scaled_X)


#plot

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.figure(figsize=(14,14))
plt.gcf()
plt.savefig("PCA.png", dpi=100)
plt.show()


#PCA with corelation



processing_data_pca= data.iloc[:, 6:135]


scaler = StandardScaler()
scaler.fit(processing_data_pca) 
scaled_processing_data_pca_no_cor = scaler.transform(processing_data_pca)



pca = PCA()
pca.fit(scaled_processing_data_pca_no_cor)
pca_data=pca.transform(scaled_processing_data_pca_no_cor)



per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.figure(figsize=(14,14))
plt.gcf()
plt.savefig("PCA.png", dpi=100)
plt.show()






#--------------------------------------------------------------------------------
#Lasso
#-------------------------------------------------------------------------------

from sklearn.linear_model import Lasso


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


prediction_lasso=lasso_regressor.predict(X)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------





