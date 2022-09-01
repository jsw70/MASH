#Workflow of the entire MASH procedure

This document is meant to be a guide for running the scripts constituting the MASH procedure detailed in ‘Method for Activity Sleep Harmonization (MASH): Harmonizing data from two wearable devices to estimate 24-hr movement cycles’  

Below is a description of each script.  The number of the script indicates the order that it should be run.  
Before running the scripts please be advised of the following:

-a)	At the beginning of every script an object is created called ‘working_dir’.  For each script this should be set to the folder in which you would like     everything to be saved/stored in.  It is assumed that the raw data files will be in this folder as well.  As you go through the steps there will be many new folders and files saved to this directory.  

a.	I would recommend downloading the entire repository on github into this folder

b)	The raw actigraph data should be stored in a hard-coded folder called ‘temp_raw_data’.  The attached scripts make this assumption.

c)	The raw actiwatch data should be stored in a hard-coded folder called ‘temp_raw_data_w_lux’.  The attached scripts make this assumption. 

d)	The diary summary data has been hardcoded as ‘diary_summary.csv’

e)	All of the Python scripts were run with version 3.9.7.  

a.	All relevant packages can be found in the ‘requirements.txt’. 

f)	Because we can’t share our data, I have provided templates of how each data table was formatted.  This can be found in the folder labelled ‘data_examples’.

g)	There are 4 main folders that are created when running all of the scripts
a.	‘raw_data_processed_’+current_date
i.	This has all of the preliminary data summaries created in steps 1 and 2
b.	‘python_batch_iteration_data’
i.	This contains folders labeled ‘set_assignment_’+current_date.  They contain all the 1D CNN elements and related objects.
c.	‘raw_data_for_1D_CNN_’+current_date
i.	This contains a subset of the raw data that is used only for building the 1D CNNs
d.	‘pred_date_’ + current_date
i.	This contains all of the objects related to generating the actual predictions.  This includes a subset of the raw data that has predictions generated. 
ii.	The final data set is saved in this folder as well.  
h)	Originally I had intended to create everything within an .RMD file however some of the python functionality wasn’t working so the scripts used are either .RMD/.NB or .py files.  It is noted where there are few instances where an .RMD file will run python code.  




Steps for running the MASH process
1)	 ‘step_1_creating_reference_for_data_status.nb’
a.	At the beginning of this script an object called ‘current_date’ is created and saved.  This is loaded in each subsequent script and is used as an identifier to separate different instances of building the MASH process (the assumption is made that it won’t happen more than once in a day!). 
b.	This loads in all of the raw data, processes it with the Choi algorithm, creates baseline activity summaries, and then categorizes each activity day with regards to the availability of sleep data.  
c.	Data categories are found in the variable ‘INTERVAL_CAT_ALT’
i.	1= there is valid sleep data preceding and proceeding the activity day
ii.	2= there is valid wake up data but not valid falling asleep data
iii.	3= there is not valid wake up data but there is valid falling asleep data
iv.	4= there is neither valid wakeup or falling asleep data

2)	‘step_2_preprocessing_data_for_the_1D_CNN_7_27_2022.nb’
a.	This processes all of the data marked with INTERVAL_CAT_ALT==1 and prepares it to be read by the data generator.  The data generator is what is used to format data to be input into the 1D CNNs
b.	There is python code embedded in this script.  These chunks were able to run through ‘reticulate’ without a problem.  These scripts convert RDS files to pickles, create the set assignments, and save the appropriate scaling objects (only after determining test, train, and validation set assignments)

3)	‘step_3a_finding_hyperparameters_with_actiwatch.py’ and ‘step_3b_finding_hyperparameters_without_actiwatch.py’
a.	These scripts run the hypberband tuner and creates a tuner object that saves the progress of the hyperparameter optimization search.  For the MASH publication each script was run for roughly 36 hours and then the python session was manually cancelled.  

4)	‘step_4a_build_cnn_with_actiwatch.py’ and ‘step_4b_build_cnn_without_actiwatch.py’
a.	These scripts load the previously created tuner objects, assign the appropriate hyperparameter values, and build the 1D CNN models.  
b.	Once the model is built a loop generates predictions for the test set.  This loop creates the test set predictions, saves them to a folder labelled ‘set_assignment_’+(the ‘current_date’ object defined in step 1), and then saves a series of plots evaluating accuracy/performance. 
 
5)	‘step_5_youden_threshold.nb’
a.	This script determines the Youden J-statistic for each model and saves objects with all the relevant information.

6)	‘step_6_building_activity_index_for_preprocessing_prediction_data.nb’
a.	This script builds something called the ‘activity_index’, which acts a reference for what the ‘INTERVAL_ CAT_ALT’ is for each person-day combination.  This index is an expansion of what was created in step 1 above because
i.	It includes people that had no valid sleep data
ii.	It imposes a ‘days per person’ restriction (this can be set at ~ line 67 using the ‘num_valid_days’ object)
b.	This script also preprocesses all of the individualized raw data to be predicted, saving the individual files in either a folder called ‘with_lux’ or ‘without_lux’.  These folders are located in the ‘pred_date_’+current_date folder that is created to save all of the prediction related objects and data. 

7)	‘step_7_generating_predictions.py’
a.	This script creates the predictions for each model.  The script contains a good amount of detail for how the ‘pred_params’ and ‘pred_params_without’ dictionaries need to be filled out.  

8)	‘step_8_creating_wake_intervals.nb’
a.	This script finishes the process of creating the wake intervals.  There are many parts to this scripts:
i.	Creating the new ‘fallasleep’ and ‘wakeup’ timestamps
ii.	Using these new timestamps to recalculate the Choi algorithm and activity measurements 
iii.	Using days categorized with ‘INTERVAL_CAT_ALT’ %in% c(“1”,”3”) to  build a bivariate probability distribution to use for estimating the time in between removing the waistband and falling asleep (as detailed in the paper this only applies to instances where the ‘fallasleep’ time is estimated using the 1D-CNN).
iv.	Calculating sleep and removing any instances where there is either still no valid data surrounding a day or a person slept for less than 60 minutes.
v.	Finalizing the dataset.  


