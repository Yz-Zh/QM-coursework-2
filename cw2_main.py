# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:49:15 2020

@author: DIY
"""

work_path='C:\\Users\\DIY\\Desktop\\cw2_processed\\QM_data.csv'
save_path_1='C:\\Users\\DIY\\Desktop\\cw2_processed\\R-F.jpg'
save_path_2='C:\\Users\\DIY\\Desktop\\cw2_processed\\outliers.txt'

#############################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf
import statsmodels.api as sms

import scipy as sp
import scipy.stats as sps

#############################

# Load the data into data frames
df = pd.read_csv(work_path)

df = df.rename(columns = {'House_prices_2011':'house_price',
                          'Burglary_rate':"burglary",
                          'Criminal_damage_rate':'criminal',
                          'Drugs_rate':'drugs',
                          'Fraud_or_Forgery_rate':'fraud_forgery',
                          'Other_Notifiable_Offences_rate':'other_offences',
                          'Robbery_rate':"robbery",
                          'Sexual_offences_rate':'sexual',
                          'Theft_and_Handling_rate':'theft',
                          'Violence_Against_the_Person_rate':'violence'})

#############################

def drop_column_using_vif_(df, thresh=5):
    while True:
        df_with_const = add_constant(df)
        vif_df = pd.Series([variance_inflation_factor(df_with_const.values, i) 
               for i in range(df_with_const.shape[1])], name= "VIF",
              index=df_with_const.columns).to_frame()
        vif_df = vif_df.drop('const') 
        if vif_df.VIF.max() > thresh:
            index_to_drop = vif_df.index[vif_df.VIF == vif_df.VIF.max()].tolist()[0]
            print('Dropping: {}'.format(index_to_drop))
            df = df.drop(columns = index_to_drop)
        else:
            break
    return df

df_new = drop_column_using_vif_(df.drop('Area', axis=1))

#############################

#regression
'''
y_values = df_new.iloc[:,1]
x_values = df_new.iloc[:,2:]
X_values = sms.add_constant(x_values)
multi_regression_model = sms.OLS(y_values, X_values).fit()
'''
multi_regression_model = smf.ols(formula='house_price ~ burglary + criminal + drugs + fraud_forgery + other_offences + robbery + theft',data=df_new).fit()

print(multi_regression_model.summary())

#############################

#size
font1={'size':17}
font2={'size':15}
font3={'size':15}
# plot R-F
plt.axhline(c="r",ls="--")
plt.scatter(multi_regression_model.fittedvalues, multi_regression_model.resid,)
#title & labels
plt.xlabel('Fitted house price',font2)
plt.ylabel('Residual',font3)
plt.title('Residual vs. Fitted Plot of house_price-crime',font1)
#save
plt.savefig(save_path_1)
plt.show()

#############################

#detect outliers
resid=list(multi_regression_model.resid)
outliers=multi_regression_model.resid.max()
print(resid.index(outliers))
#save
'''with open(save_path_2,'w') as save_object:
    save_object.writelines(df.iloc[0,resid.index(outliers)])
    save_object.close()
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    