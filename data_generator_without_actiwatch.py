import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.utils import Sequence
import os 
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import pickle
import joblib





class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_IDs, shuffle=True,lowerbound=0,upperbound=0,date_folder='A',working_directory='A',
    build_vars=['A'],
    explode_vars=['A'],cur_date='A'):
        #'Initialization'
        self.date_folder=date_folder
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.lowerbound=lowerbound
        self.upperbound=upperbound
        self.build_vars=build_vars
        self.explode_vars=explode_vars
        self.working_directory=working_directory
        self.cur_date=cur_date
        self.on_epoch_end()

    ###############################################

    
    def on_epoch_end(self):
    #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    ################################################
    def col_explode_fct(self,df,actvar,lowerbound,upperbound):
        dat=df[['ID','TimeStamp',actvar,'REAL_DATE_EXT']].copy()

    # for x in range(lowerbound, upperbound+1):
            #dat.loc[:,x] = dat.groupby(['ID','REAL_DATE_EXT'])[actvar].shift(x)
        for x in range(lowerbound, upperbound+1):
            dat= pd.concat([dat,pd.DataFrame({str(x):dat.groupby(['ID','REAL_DATE_EXT'])[actvar].shift(x)})],axis=1)
        
        dat=dat.drop([actvar],axis='columns').dropna()

        dat=pd.melt(dat,id_vars=['ID','REAL_DATE_EXT','TimeStamp'])

        dat.loc[:,'variable']=dat.variable.astype(int)  

        dat=dat.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

        return dat.rename(columns={"value":actvar})

    ################################################

    def __data_generation(self, list_IDs_temp):
    #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        dat=pd.read_pickle(self.working_directory+'raw_data_for_1D_CNN_'+self.cur_date+'/pickles/'+list_IDs_temp+'.pkl')

        scale_path=self.working_directory+'/python_batch_iteration_data/'+self.date_folder+'/'
        
        mm_axis1_scaler=joblib.load(scale_path+'mm_axis1_scaler.save')

        mm_axis2_scaler=joblib.load(scale_path+'mm_axis2_scaler.save')

        mm_axis3_scaler=joblib.load(scale_path+'mm_axis3_scaler.save')

        dat.loc[:,'WI']=np.where((dat.TimeStamp>=dat.PREV_END) & (dat.TimeStamp<=dat.SLPITSRT15), 1,0)
        
        dat.loc[:,'wearing']=np.where(dat.wearing=="w",1,0)

        #dat.loc[:,'prev_wearing']=dat.wearing.shift(periods=-1,axis=0)

        dat=dat.sort_values(['ID','TimeStamp'],ascending=True)

        total_dates=dat.REAL_DATE_EXT.unique()

        if len(total_dates)<=2 :
            possible_dates=total_dates
        else: 
            possible_dates=np.random.choice(a=dat.REAL_DATE_EXT.unique(),size=2,replace=False)

        dat=dat[dat.REAL_DATE_EXT.isin(possible_dates)].sort_values(['ID','REAL_DATE_EXT','TimeStamp'])

        dat.loc[:,'WEIGHT']=np.where(((dat.TimeStamp>=(dat.PREV_END-pd.Timedelta(minutes=120))) & 
                                        (dat.TimeStamp<=(dat.PREV_END+pd.Timedelta(minutes=120)))) |
                                       ((dat.TimeStamp>=(dat.SLPITSRT15-pd.Timedelta(minutes=120))) & 
                                        (dat.TimeStamp<=(dat.SLPITSRT15+pd.Timedelta(minutes=120)))) ,15,1)

        dat.loc[:,'WEIGHT']=dat.groupby(['ID','TimeStamp'])['WEIGHT'].transform('max')

        dat.loc[:,'Axis1']=mm_axis1_scaler.transform(dat[['Axis1']]).reshape(dat.shape[0])
        dat.loc[:,'Axis2']=mm_axis2_scaler.transform(dat[['Axis2']]).reshape(dat.shape[0])
        dat.loc[:,'Axis3']=mm_axis3_scaler.transform(dat[['Axis3']]).reshape(dat.shape[0])



        ######################################################################
        counter=0
        for j in self.explode_vars:
            counter+=1
            dataxtemp=self.col_explode_fct(df=dat,actvar=j,lowerbound=self.lowerbound,upperbound=self.upperbound)
            if counter>1:
                datax1=pd.merge(datax1,dataxtemp,how='inner',on=['ID','REAL_DATE_EXT','TimeStamp','variable'])
            else: 
                datax1=dataxtemp.copy()

#########################################################################
########################################################################
###############################################################################

        datax1=datax1.dropna()

        dat=pd.merge(dat,datax1[['TimeStamp']].drop_duplicates(),how='inner',on='TimeStamp')

        sampler=dat[['TimeStamp','REAL_DATE_EXT','WEIGHT','WI','PREV_END','SLPITSRT15']].drop_duplicates()

        dat=1

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

        sampler=sampler.sample(n=512,ignore_index=True,weights=sampler.WEIGHT,replace=False)

        datax1=pd.merge(datax1,sampler,on=['TimeStamp','REAL_DATE_EXT'],how='inner').drop_duplicates()

        datax1=datax1.sort_values(['ID','REAL_DATE_EXT','TimeStamp','variable'],ascending=[True,True,True,False])

        timesteps=len(datax1.variable.unique())
        ####
        tag_id=datax1[['REAL_DATE_EXT','TimeStamp']].drop_duplicates().shape[0]

        X=datax1[self.build_vars].values.reshape(tag_id,timesteps,len(self.build_vars))
        y1=datax1[datax1.variable==0][['WI']].values
        Y=[y1]

        return X,Y

    ######################
    def __getitem__(self, index):
    #'Generate one batch of data'
    # Generate indexes of the batch
    # Find list of IDs
        list_IDs_temp = self.list_IDs[index] 

    # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __len__(self):
  #'Denotes the number of batches per epoch'
        return len(self.list_IDs) 


