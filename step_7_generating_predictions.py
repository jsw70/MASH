
import pandas as pd
import stats
import sklearn
import numpy as np
import datetime
from datetime import date
import numpy.lib.stride_tricks as stride
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os 
import stats
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from importlib import reload 
import pickle
import datetime
import pathlib
import joblib
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from os import listdir
from os.path import isfile, join
from datetime import date

working_dir="L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/"
os.chdir(working_dir + 'python_batch_iteration_data/')


current_date=pd.read_csv(working_dir + 'current_date_reference_tag.csv')['x'].iloc[0]
##model reference folder: this is the folder where all of the model building data is found
##scalar name: this is the name of the scaling object used to process the input data
##threshold_filename:  This is the file containing the optimal cutoff point of the probability prediction
#### parent folder: L:/SWAN/data analysis/Analyst Folders/JF/projects/Colvin_5_21_2021/actigraphy/data/model_building/python_batch_iteration_data

##data reference folder: this is where the data that is in need of 'correction' can be found
### parent folder: L:/SWAN/data analysis/Analyst Folders/JF/projects/Colvin_5_21_2021/actigraphy/data/

##explode_vars: these are the variables that get processed in 'long-form' according to the 'col_explode_fct' defined below

##total_vars: this is a list of the total input variables to the model

##lowerbound and upperbound: this determines how far to look backwards and forwards for each epoch
# ###this should be a copy of what was used to build the model otherwise you won't be able to predict anything
# ### because the data will have the wrong dimensions (or, if predictions are created it will just be wrong in general!)
 

pred_params={'model_reference_folder':'set_assignment_'+current_date,
'model_name':'1D_CNN_WITH_LUX'+current_date+'.h5',
'data_reference_folder':'pred_date_'+current_date+'/with_lux',
'explode_vars':['Axis1','Axis2','Axis3','WhiteLight','Activity','wearing'],
'total_vars':['Axis1','Axis2','Axis3','WhiteLight','Activity','wearing','AFTERNOON','NIGHT'],
'lowerbound':-50,
'upperbound':50
}

pred_params_without={'model_reference_folder':'set_assignment_'+current_date,
'model_name':'1D_CNN_WITHOUT_LUX'+current_date+'.h5',
'data_reference_folder':'pred_date_'+current_date+'/without_lux',
'explode_vars':['Axis1','Axis2','Axis3','wearing'],
'total_vars':['Axis1','Axis2','Axis3','wearing','AFTERNOON','NIGHT'],
'lowerbound':-50,
'upperbound':50
}




#######DONT FORGET TO SET THE PARAMS ACCORDINGLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###REFER TO THE DOCUMENTATION FOR GUIDANCE
def col_explode_fct(df,actvar,lowerbound,upperbound):
    dat=df[['ID','TimeStamp',actvar,'REAL_DATE_EXT']].copy()

# for x in range(lowerbound, upperbound+1):
        #dat.loc[:,x] = dat.groupby(['ID','REAL_DATE_EXT'])[actvar].shift(x)
    for x in range(lowerbound, upperbound+1):
        dat= pd.concat([dat,pd.DataFrame({str(x):dat.groupby(['ID','REAL_DATE_EXT'])[actvar].shift(x)})],axis="columns")
    
    dat=dat.drop([actvar],axis='columns').dropna()

    dat=pd.melt(dat,id_vars=['ID','REAL_DATE_EXT','TimeStamp'])

    dat.loc[:,'variable']=dat.variable.astype(int)  

    dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

    return dat.rename(columns={"value":actvar})



##load the model for 'with_lux'

model=keras.models.load_model(working_dir+'python_batch_iteration_data/'+pred_params['model_reference_folder']+'/'+pred_params['model_name'])

#threshold for 'with_lux' prediction
# threshold=pd.read_pickle('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/model_building/python_batch_iteration_data/'+pred_params['model_reference_folder'] +'/'+pred_params['threshold_filename'])

# threshold=threshold.YOUDEN_CUTOFF.unique()[0]


#######################################################
###########WITH LUX
#this loop will generate the predictions and save them to a dataframe called 'predictions'
# ############################################################

os.chdir(working_dir+pred_params['data_reference_folder'])

#get all ids to make predictions (with lux)
ids = [f for f in listdir('pickles/') if isfile(join('pickles/', f))]


predictions_awake=pd.DataFrame()

predictions_asleep=pd.DataFrame()

id_df=pd.DataFrame()

import warnings

with warnings.catch_warnings():
    #this turns warnings into an error, so if any warnings happen the process will stop and we can investigate
    warnings.simplefilter('error')
    for i in ids:
        #load and sort data
        dat=pd.read_pickle('pickles/'+i)

        dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp'])
        
        dat.loc[:,'wearing']=np.where(dat.wearing=='w',1,0)

        scale_path1=working_dir + '/python_batch_iteration_data/set_assignment_'+current_date+'/'


        mm_axis1_scaler=joblib.load(scale_path1+'mm_axis1_scaler.save')

        mm_axis2_scaler=joblib.load(scale_path1+'mm_axis2_scaler.save')

        mm_axis3_scaler=joblib.load(scale_path1+'mm_axis3_scaler.save')

        mm_activity_scaler=joblib.load(scale_path1+'mm_activity_scaler.save')

        mm_whitelight_scaler=joblib.load(scale_path1+'mm_whitelight_scaler.save')

        dat.loc[:,'Axis1']=mm_axis1_scaler.transform(dat[['Axis1']]).reshape(dat.shape[0])
        dat.loc[:,'Axis2']=mm_axis2_scaler.transform(dat[['Axis2']]).reshape(dat.shape[0])
        dat.loc[:,'Axis3']=mm_axis3_scaler.transform(dat[['Axis3']]).reshape(dat.shape[0])
        dat.loc[:,'WhiteLight']=mm_whitelight_scaler.transform(dat[['WhiteLight']]).reshape(dat.shape[0])
        dat.loc[:,'Activity']=mm_activity_scaler.transform(dat[['Activity']]).reshape(dat.shape[0])

        #iterate over 'explode vars' to create the 'datax1' dataset that will be the final thing fed into the model
        counter=0
        for j in pred_params['explode_vars']:
            counter+=1
            dataxtemp=col_explode_fct(df=dat,actvar=j,lowerbound=pred_params['lowerbound'],upperbound=pred_params['upperbound'])
            if counter>1:
                datax1=pd.merge(datax1,dataxtemp,how='inner',on=['ID','REAL_DATE_EXT','TimeStamp','variable'])
            else: 
                datax1=dataxtemp.copy()

        #this is a dataset for the 'fixed' varaibles 
        sampler=dat[['TimeStamp','REAL_DATE_EXT']].drop_duplicates()

        sampler.loc[:,'HOUR']=sampler['TimeStamp'].dt.hour

        conditions = [(sampler.HOUR>=22) | (sampler.HOUR<=6),
        (sampler.HOUR>6) & (sampler.HOUR<=13),
        (sampler.HOUR>13) & (sampler.HOUR<22)]

        choices = ['NIGHT','MORNING','AFTERNOON']

        sampler.loc[:,'HOUR'] = np.select(conditions, choices, default=np.nan)

        #sampler.loc[:,'wearing']=np.where(sampler.wearing=="w",1,0)

        he=pd.get_dummies(sampler.HOUR)

        sampler=sampler.drop(['HOUR'],axis='columns')

        sampler=sampler.join(he)

        sampler=sampler.drop('MORNING',axis='columns')

        datax1=pd.merge(datax1,sampler,on=['TimeStamp','REAL_DATE_EXT'],how='inner').drop_duplicates()

        datax1=datax1.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

        timesteps=len(datax1.variable.unique())

        datax1=datax1.dropna()

        id_df=id_df.append(pd.merge(dat,datax1[['ID','REAL_DATE_EXT','TimeStamp']].drop_duplicates(),
        how='inner',
        on=['ID','REAL_DATE_EXT','TimeStamp']))

        dat=1

        tag_id=datax1[['TimeStamp','REAL_DATE_EXT']].drop_duplicates().shape[0]

        predictions_awake=predictions_awake.append(pd.DataFrame(model.predict(datax1[pred_params['total_vars']].values.reshape(tag_id,timesteps,len(pred_params['total_vars'])))))

        # predictions_asleep=predictions_asleep.append(pd.DataFrame(model.predict(datax1[pred_params['total_vars']].values.reshape(tag_id,timesteps,len(pred_params['total_vars'])))[2]))
        
        if predictions_awake.shape[0]!=id_df.shape[0]:
            raise Exception('predictiosn not equal to id_df')


##here we complete the prediction dataset by renaming the prediction column and joining more characteristics
predictions_awake=predictions_awake.rename( columns={0: "PRED_AWAKE"})
predictions_awake.reset_index(inplace=True,drop=True)
id_df.reset_index(inplace=True,drop=True)
pred_w_lux=pd.concat([id_df,predictions_awake],axis=1)
pred_w_lux.loc[:,'PREDICTION_CATEGORY']='WITH_LUX_DATA'

pred_w_lux.to_pickle(working_dir + pred_params['data_reference_folder'][0:pred_params['data_reference_folder'].find('/')]+'/pred_w_lux_SINGLE_' + current_date+'.pkl')


##########################################################################
##WITHOUT LUX
##########################################################################
##load the model for 'with_lux'
os.chdir(working_dir+pred_params_without['data_reference_folder'])

model2=keras.models.load_model(working_dir+'python_batch_iteration_data/'+pred_params_without['model_reference_folder']+'/'+pred_params_without['model_name'])

# threshold_without_lux=pd.read_pickle('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/model_building/python_batch_iteration_data/'+pred_params_without['model_reference_folder'] +'/'+pred_params_without['threshold_filename'])

# threshold_without_lux=threshold_without_lux.YOUDEN_CUTOFF.unique()[0]

ids = [f for f in listdir('pickles/') if isfile(join('pickles/', f))]

predictions_without_lux=pd.DataFrame()

id_df_without_lux=pd.DataFrame()


for i in ids:

    dat=pd.read_pickle('pickles/'+i)

    dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp'])

    dat.loc[:,'wearing']=np.where(dat.wearing=='w',1,0)

    scale_path1=working_dir+'/python_batch_iteration_data/set_assignment_'+current_date+'/'

    mm_axis1_scaler=joblib.load(scale_path1+'mm_axis1_scaler.save')

    mm_axis2_scaler=joblib.load(scale_path1+'mm_axis2_scaler.save')

    mm_axis3_scaler=joblib.load(scale_path1+'mm_axis3_scaler.save')

    dat.loc[:,'Axis1']=mm_axis1_scaler.transform(dat[['Axis1']]).reshape(dat.shape[0])
    dat.loc[:,'Axis2']=mm_axis2_scaler.transform(dat[['Axis2']]).reshape(dat.shape[0])
    dat.loc[:,'Axis3']=mm_axis3_scaler.transform(dat[['Axis3']]).reshape(dat.shape[0])
        

    counter=0
    for j in pred_params_without['explode_vars']:
        counter+=1
        dataxtemp=col_explode_fct(df=dat.copy(),actvar=j,lowerbound=pred_params_without['lowerbound'],upperbound=pred_params_without['upperbound'])
        if counter>1:
            datax1=pd.merge(datax1,dataxtemp,how='inner',on=['ID','REAL_DATE_EXT','TimeStamp','variable'])
        else: 
            datax1=dataxtemp.copy()


    sampler=dat[['TimeStamp','REAL_DATE_EXT']].drop_duplicates()

    sampler.loc[:,'HOUR']=sampler['TimeStamp'].dt.hour

    conditions = [(sampler.HOUR>=22) | (sampler.HOUR<=6),
    (sampler.HOUR>6) & (sampler.HOUR<=13),
    (sampler.HOUR>13) & (sampler.HOUR<22)]

    choices = ['NIGHT','MORNING','AFTERNOON']

    sampler.loc[:,'HOUR'] = np.select(conditions, choices, default=np.nan)

    he=pd.get_dummies(sampler.HOUR)

    sampler=sampler.drop(['HOUR'],axis='columns')

    sampler=sampler.join(he)

    sampler=sampler.drop('MORNING',axis='columns')

    datax1=pd.merge(datax1,sampler,on=['TimeStamp','REAL_DATE_EXT'],how='inner').drop_duplicates()

    datax1=datax1.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

    timesteps=len(datax1.variable.unique())

    datax1=datax1.dropna()

    id_df_without_lux=id_df_without_lux.append(pd.merge(dat,datax1[['ID','REAL_DATE_EXT','TimeStamp']].drop_duplicates(),
    how='inner',
    on=['ID','REAL_DATE_EXT','TimeStamp']))

    dat=1

    tag_id=datax1[['TimeStamp','REAL_DATE_EXT']].drop_duplicates().shape[0]

    predictions_without_lux=predictions_without_lux.append(pd.DataFrame(model2.predict(datax1[pred_params_without['total_vars']].values.reshape(tag_id,timesteps,len(pred_params_without['total_vars'])))))

    if predictions_without_lux.shape[0]!=id_df_without_lux.shape[0]:
        raise Exception('predictiosn not equal to id_df_without_lux')

predictions_without_lux=predictions_without_lux.rename( columns={0: "PRED_AWAKE"})
predictions_without_lux.reset_index(inplace=True,drop=True)
id_df_without_lux.reset_index(inplace=True,drop=True)
pred_wo_lux=pd.concat([id_df_without_lux,predictions_without_lux],axis=1)
pred_wo_lux.loc[:,'PREDICTION_CATEGORY']='WITHOUT_LUX_DATA'
# pred_wo_lux.loc[:,'PRED_CLASS']=np.where(pred_wo_lux.PRED_AWAKE>=threshold_without_lux,1,0)


#pd.merge(dat,datax1[['ID','REAL_DATE_EXT','TimeStamp']].drop_duplicates(), how='inner',on=['ID','REAL_DATE_EXT','TimeStamp']).shape
################################################################################
#################################################################################


os.chdir(working_dir+pred_params['data_reference_folder'][0:pred_params['data_reference_folder'].find('/')])


pred_w_lux.reset_index(inplace=True,drop=True)
pred_wo_lux.reset_index(inplace=True,drop=True)

final_predictions=pd.concat([pred_w_lux,pred_wo_lux],axis=0)

final_predictions.reset_index(inplace=True,drop=True)

final_predictions.loc[:,'REAL_DATE_EXT_ALT']=final_predictions.REAL_DATE_EXT.astype('float')

final_predictions.to_pickle('final_predictions_'+current_date+'.pkl')




