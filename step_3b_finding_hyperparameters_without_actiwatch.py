
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

working_dir="enter working directory here"

os.chdir(working_dir + 'python_batch_iteration_data/')

current_date=pd.read_csv(working_dir + 'current_date_reference_tag.csv')['x'].iloc[0]

sys.path.append(working_dir)

from data_generator_without_actiwatch import DataGenerator

params = {
        'shuffle': True,
        'lowerbound':-50,
        'upperbound':50,
        'date_folder':'set_assignment_'+current_date,
        'build_vars':['Axis1','Axis2','Axis3','wearing','AFTERNOON','NIGHT'],
        'explode_vars':['Axis1','Axis2','Axis3','wearing'],
        'working_directory':working_dir,
        'cur_date':current_date}

###############################################################################################
###############################################################################################

os.chdir(working_dir + 'python_batch_iteration_data/set_assignment_'+current_date+'/')

#exclude_tags=np.load('exclude_tags.npy',allow_pickle=True)

#####################################################################################33
######################################################################################################

id_df=pd.read_pickle('sa_'+current_date+'.pkl')

partition={'train':id_df[(id_df.GROUP=='TRAIN') ].ID.values,
'val':id_df[(id_df.GROUP=='VAL') ].ID.values,
'test':id_df[(id_df.GROUP=='TEST') ].ID.values}

training_generator = DataGenerator(partition['train'], **params)

validation_generator = DataGenerator(partition['val'], **params)

######################################################################################################################
###################################################################################################################

def building_model(hp):
    ks1_units = hp.Choice('ks1', values=[2,3,5,7,11,13])
    f1 = hp.Choice('f1', values=[16,32,64,128,252])
    ks2_units = hp.Choice('ks2', values=[2,3,5,7,11,13])
    f2 = hp.Choice('f2', values=[16,32,64,128,252])
    d3 = hp.Choice('d3', values=[25, 50,75,100])
    d4 = hp.Choice('d4', values=[5, 10,15,20])
    do1 = hp.Choice('do1', values=[0.2,0.3,0.4,0.5])
    do2 = hp.Choice('do2', values=[0.2,0.3,0.4,0.5])

    inp=Input(shape=[params['upperbound']-params['lowerbound']+1,len(params['build_vars'])])
    x1= Conv1D(filters=f1,kernel_size=ks1_units,padding='same',use_bias=True,activation='relu')(inp)
    x1=MaxPooling1D(pool_size=2)(x1)
    x5=Conv1D(filters=f2,kernel_size=ks2_units,use_bias=True,padding='same',activation='relu')(x1)
    x5=Flatten()(x5)
    x5=Dense(d3,activation='relu')(x5)
    x5=Dropout(do1)(x5)
    x5=Dense(d4,activation='relu')(x5)
    x5=Dropout(do2)(x5)

    y_pred1 = Dense(1, activation='sigmoid', name = 'wake_interval')(x5)
    
    model=Model(inputs=inp, outputs=[y_pred1])

    model.compile(loss= 'binary_crossentropy',
              optimizer='Adam',
              metrics = [tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()])

    return model

##########################################################################
#########################################################################

tuner = kt.Hyperband(building_model,
                     objective=kt.Objective('val_loss',direction='min'),
                     max_epochs=25,
                     factor=3,
                     directory='my_dir',
                     project_name='hyp_single_without_lux')


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(training_generator, validation_data=validation_generator,use_multiprocessing=True,workers=6, callbacks=[stop_early],verbose=1)




# Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters()[0]









