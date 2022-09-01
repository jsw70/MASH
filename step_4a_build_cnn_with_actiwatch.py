
import pandas as pd
import stats
import sklearn
import numpy as np
import datetime
from datetime import date
import numpy.lib.stride_tricks as stride
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from importlib import reload 
import pickle
import datetime
import pathlib
import joblib
import tensorboard
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import keras_tuner as kt

working_dir="L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/"
os.chdir(working_dir + 'python_batch_iteration_data/')


current_date=pd.read_csv(working_dir + 'current_date_reference_tag.csv')['x'].iloc[0]

sys.path.append(working_dir)

from data_generator_with_actiwatch import DataGenerator


#########################################################################################################################
#########################################################################################################################

##this function creates the 101-minute (epoch) windows used as inputs
def col_explode_fct(df,actvar,lowerbound,upperbound):
    dat=df[['ID','TimeStamp',actvar,'REAL_DATE_EXT']].copy()

    for x in range(lowerbound, upperbound+1):
        dat= pd.concat([dat,pd.DataFrame({str(x):dat.groupby(['ID','REAL_DATE_EXT'])[actvar].shift(x)})],axis=1)
    
    dat=dat.drop([actvar],axis='columns').dropna()

    dat=pd.melt(dat,id_vars=['ID','REAL_DATE_EXT','TimeStamp'])

    dat.loc[:,'variable']=dat.variable.astype(int)  

    dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

    return dat.rename(columns={"value":actvar})



###############################################################################################
###############################################################################################

os.chdir(working_dir+'/python_batch_iteration_data/set_assignment_'+current_date+'/')

#exclude_tags=np.load('exclude_tags.npy',allow_pickle=True)

params = {
        'shuffle': True,
        'lowerbound':-50,
        'upperbound':50,
        'date_folder':'set_assignment_'+current_date,
        'build_vars':['Axis1','Axis2','Axis3','WhiteLight','Activity','wearing','AFTERNOON','NIGHT'],
        'explode_vars':['Axis1','Axis2','Axis3','WhiteLight','Activity','wearing'],
        'working_directory':working_dir}

#####################################################################################33
######################################################################################################

id_df=pd.read_pickle('sa_'+current_date+'.pkl')

partition={'train':id_df[(id_df.GROUP=='TRAIN') ].ID.values,
'val':id_df[(id_df.GROUP=='VAL') ].ID.values,
'test':id_df[(id_df.GROUP=='TEST') ].ID.values}

training_generator = DataGenerator(partition['train'], **params)

validation_generator = DataGenerator(partition['val'], **params)

################################################################################################
################################################################################################
#############################################################################################################

################################################################################################################
################################################################################################################

##load the previously created tuner and get the best hyperparameters

tuner = kt.Hyperband(
                     objective=kt.Objective('val_loss',direction='min'),
                     max_epochs=25,
                     factor=3,
                     directory='my_dir',
                     project_name='hyp_single_w_lux')

best_hps=tuner.get_best_hyperparameters()[0]


model_params={'ks1':best_hps['ks1'],
'ks2':best_hps['ks2'],
'f1':best_hps['f1'],
'f2':best_hps['f2'],
'd3':best_hps['d3'],
'd4':best_hps['d4'],
'do1':best_hps['do1'],
'do2':best_hps['do2'],
'name':'1D_CNN_WITH_LUX'}


##build the model
inp=Input(shape=[params['upperbound']-params['lowerbound']+1,len(params['build_vars'])])
x1= Conv1D(filters=model_params['f1'],kernel_size=model_params['ks1'],padding='same',use_bias=True,activation='relu')(inp)
x1=MaxPooling1D(pool_size=2)(x1)
x5=Conv1D(filters=model_params['f2'],kernel_size=model_params['ks2'],use_bias=True,padding='same',activation='relu')(x1)
x5=Flatten()(x5)
x5=Dense(model_params['d3'],activation='relu')(x5)
x5=Dropout(model_params['do2'])(x5)
x5=Dense(model_params['d4'],activation='relu')(x5)
x5=Dropout(model_params['do2'])(x5)

y1_pred = Dense(1, activation='sigmoid', name = 'wake_interval')(x5)


model=Model(inputs=inp, outputs=[y1_pred])

model.compile(loss=[ 'binary_crossentropy'],
optimizer='Adam',
              metrics = {'wake_interval': [tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy(),
              tf.keras.metrics.Precision()]})

model.summary()

#############################################################################################
##########################################################################################
##########################################################################################

history=model.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=1,
                    verbose=1)

#############################generate predictions for the test set

model.save(model_params['name']+current_date  + '.h5')
test_pred=pd.DataFrame()
test_answers=pd.DataFrame()

for i in partition['test']:

    dat=pd.read_pickle(working_dir + 'raw_data_for_1D_CNN_'+current_date+'/pickles/'+i+'.pkl')

    scale_path=working_dir + 'python_batch_iteration_data/set_assignment_'+current_date+'/'
    
    mm_axis1_scaler=joblib.load(scale_path+'mm_axis1_scaler.save')

    mm_axis2_scaler=joblib.load(scale_path+'mm_axis2_scaler.save')

    mm_axis3_scaler=joblib.load(scale_path+'mm_axis3_scaler.save')

    mm_activity_scaler=joblib.load(scale_path+'mm_activity_scaler.save')

    mm_whitelight_scaler=joblib.load(scale_path+'mm_whitelight_scaler.save')
    
    dat.loc[:,'WI']=np.where((dat.TimeStamp>=dat.PREV_END) & (dat.TimeStamp<=dat.SLPITSRT15), 1,0)

    dat.loc[:,'wearing']=np.where(dat.wearing=='w',1,0)

    dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp'])

    dat.loc[:,'Axis1']=mm_axis1_scaler.transform(dat[['Axis1']]).reshape(dat.shape[0])
    dat.loc[:,'Axis2']=mm_axis2_scaler.transform(dat[['Axis2']]).reshape(dat.shape[0])
    dat.loc[:,'Axis3']=mm_axis3_scaler.transform(dat[['Axis3']]).reshape(dat.shape[0])
    dat.loc[:,'WhiteLight']=mm_whitelight_scaler.transform(dat[['WhiteLight']]).reshape(dat.shape[0])
    dat.loc[:,'Activity']=mm_activity_scaler.transform(dat[['Activity']]).reshape(dat.shape[0])


    counter=0
    for j in params['explode_vars']:
        counter+=1
        dataxtemp=col_explode_fct(df=dat,actvar=j,lowerbound=params['lowerbound'],upperbound=params['upperbound'])
        if counter>1:
            datax1=pd.merge(datax1,dataxtemp,how='inner',on=['ID','REAL_DATE_EXT','TimeStamp','variable'])
        else: 
            datax1=dataxtemp.copy()
    ###################################################################

    dataxtemp=1

    datax1=pd.merge(datax1,dat[['ID','REAL_DATE_EXT','TimeStamp','WI']],how='inner',on=['ID','REAL_DATE_EXT','TimeStamp'])

    dat=1

    datax1.loc[:,'HOUR']=datax1['TimeStamp'].dt.hour

    conditions = [(datax1.HOUR>=22) | (datax1.HOUR<=6),
    (datax1.HOUR>6) & (datax1.HOUR<=13),
    (datax1.HOUR>13) & (datax1.HOUR<22)]

    choices = ['NIGHT','MORNING','AFTERNOON']

    datax1.loc[:,'HOUR'] = np.select(conditions, choices, default=np.nan)

    he=pd.get_dummies(datax1.HOUR)

    datax1=datax1.drop('HOUR',axis='columns')

    datax1=datax1.join(he)

    datax1=datax1.drop('MORNING',axis='columns')

    timesteps=len(datax1.variable.unique())

    datax1=datax1.dropna()

    tag_id=datax1[['TimeStamp','REAL_DATE_EXT']].drop_duplicates().shape[0]

    test_pred=test_pred.append(pd.DataFrame(model.predict(datax1[params['build_vars']].values.reshape(tag_id,timesteps,len(params['build_vars'])))))
    test_answers=test_answers.append(datax1[datax1.variable==0][['ID','TimeStamp','WI','REAL_DATE_EXT','wearing']])


test_pred=test_pred.rename( columns={0: "PRED_AWAKE"})

test_answers.reset_index(inplace=True,drop=True)
test_pred.reset_index(inplace=True,drop=True)
test_pred=pd.concat([test_answers,test_pred],axis=1)

#################################################    
########save the final predictions
test_pred.to_pickle(working_dir+'python_batch_iteration_data/set_assignment_'+current_date+'/'+'predictions_'+model_params['name']+'.pkl')

####create some informative plots#
####################################
fpr, tpr, _ = roc_curve(test_answers.loc[:, 'WI'],test_pred.loc[:, 'PRED_AWAKE'])
roc_auc= auc(fpr, tpr)
#
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.3, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_AUC'+model_params['name'] +'.png')
#plt.show()

#########################################################################

plt.figure() 
plt.plot(model.history['binary_accuracy'])
plt.plot(model.history['val_binary_accuracy'])
plt.title('model binary accuracy, threshold=0.5')
plt.ylabel('binary accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Acc_epoch'+model_params['name'] +'.png')
#plt.show()
# summarize history for loss

#################################################################

lr_precision, lr_recall, _ = precision_recall_curve(test_answers.loc[:,'WI'], test_pred.loc[:,'PRED_AWAKE'])
lr_auc =  auc(lr_recall, lr_precision)
# summarize scores
# plot the precision-recall curves
no_skill = len(test_answers[test_answers.WI==1].loc[:, 'WI']) / test_answers.shape[0]
plt.figure()
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No model')
plt.plot(lr_recall, lr_precision, marker='.', label='cnn')
plt.title('CNN:  auc=%.3f' % ( lr_auc))
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.savefig('PR_CURVE' +model_params['name'] +'.png')
#plt.show()

###############################################################
plt.figure()
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('Loss_epoch' +model_params['name'] +'.png')
#plt.show()



################################################################################################################
################################################################################################################
