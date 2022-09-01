library(tidyverse)
library(stringr)
library(lubridate)
library(PhysicalActivity)
library(ggplot2)
library(data.table)
library(RcppRoll)
library(zoo)
library(foreign)
library(haven)
options(dplyr.width = Inf)
library(imputeTS)

current_date=date(Sys.time())


#########################################

##path to all of the actigraphy files
folder_loc="L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/raw_data/temp_raw_data/"


folder_name=substr(folder_loc,
                   data.frame(str_locate_all(folder_loc,"/")[[1]])$end[length(data.frame(str_locate_all(folder_loc,"/")[[1]])$end)-1]+1,
                   data.frame(str_locate_all(folder_loc,"/")[[1]])$end[length(data.frame(str_locate_all(folder_loc,"/")[[1]])$end)]-1)

files=list.files(folder_loc,pattern=".csv")

##########################################
##############################################

folder_loc_lux="L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/raw_data/temp_raw_data_w_lux/"

folder_name_lux=substr(folder_loc_lux,
                       data.frame(str_locate_all(folder_loc_lux,"/")[[1]])$end[length(data.frame(str_locate_all(folder_loc_lux,"/")[[1]])$end)-1]+1,
                       data.frame(str_locate_all(folder_loc_lux,"/")[[1]])$end[length(data.frame(str_locate_all(folder_loc_lux,"/")[[1]])$end)]-1)

#get a list of files
files_lux=list.files(folder_loc_lux,pattern=".csv")


############################################
##############################################

letter_space="[[abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ]]"

diary_dates=tibble(read.csv("L:/SWAN/data management/Actigraph/Activity/DATA/CSV/SPAD/spad_ALL_201705083.csv"))

source("L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/raw_data_functions_for_github_6_23_2022.R")

source("L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/header_script_for_github_7_13_2022.R")


header_dat=do.call("rbind",
                   lapply(seq(from=1,
                              to=length(files),
                              by=1),
                          header_fct))

header_dat=header_dat %>% left_join(y=diary_dates %>% 
                                      mutate(ID=as.character(ID),
                                             across(c("STARTWAT","STOPWAT","STARTMON","STOPMON"), ~as.Date(.x,format="%m/%d/%Y"))) %>%
                                      select(ID,VISIT,STARTWAT,STOPWAT,STARTMON,STOPMON,TRNDT,SITE_ID),
                                    by="ID")



#########################################################################################################################################
######THIS COMPLETES PART 1 OF THE DOCUMENTATION ########################################################################################
#########################################################################################################################################

######creating the base activity data
#value for pre_filter
####TRUE indicates that you want the raw data truncated within monitor-wearing dates before choi algorithm is run
####FALSE that you don't
###################
preff=TRUE

activity_data=suppressWarnings(do.call("rbind",lapply(seq(from=1,
                                     to=length(files),
                                     by=1),
                                 choi_process,
                                 prefilter=preff)))


activity_data_name=paste(length(unique(activity_data$ID)),
"_from_",
folder_name,
"_",
str_replace_all(current_date,"-","_"),
"_prefilter_",
preff,
".Rda",sep="")

##save the activity data if necessary

saveRDS(activity_data,
        file=paste('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/JF_activity_data/',activity_data_name,sep=""))


##########################################################################################################################################
##################################################################################################################################################################
##THIS COMPLETES PART 2 OF THE DOCUMENTATION 
#############################################################################
#############################################################################

#activity_data_name was set in the section above
##output is named 'combined_days_final' in console
###output is saved as:
 paste('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/raw_data/JF_processed/tier_1_done/TIER_1_',current_date,
      '.Rda',sep="")

activity_data_name="1322_from_temp_raw_data_2022_05_05_prefilter_TRUE.Rda"

source("L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/initial_day_night_join_github_7_14_2022.R")


#########################################################################################################################################
######THIS COMPLETES PART 5 OF THE DOCUMENTATION ########################################################################################
#########################################################################################################################################

##this is a process applied to the NON-raw daily activity and sleep data

interval_dat=tibble(readRDS( paste('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/raw_data/JF_processed/tier_1_done/TIER_1_',current_date,
                                   '.Rda',sep="")))

interval_dat = interval_dat  %>% mutate(REAL_DATE=as.Date(STRDATE_ALT,format="%Y-%m-%d"))
########################################################

##load the activity data created earlier in this script
##only do this if you haven't run the earlier parts of the script in this session
# activity_data=tibble(readRDS('L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/JF_activity_data/1322_from_temp_raw_data_2022_03_21_prefilter_TRUE.Rda'))

#####################
##create a new folder to store the data for predictions in
mainDir="L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/data/data_for_prediction/"
subDir=paste("pred_date",current_date,sep="_")
dir.create(file.path(mainDir, subDir))
setwd(file.path(mainDir, subDir))
dir.create(file.path("with_lux"))
dir.create(file.path("without_lux"))
#########################

#####################################################
#########input needed################################
###the variable 'num_valid_days' indicates what the lowest number of days total per person allowed are
num_valid_days=4
 
source("L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/raw_process_console_building_activity_index_2_21_2022.R")

##these are created in the above source() command

###cat_type_count_df has a break down of the interval categories
cat_type_count_df

###activity_index is a list to determine who does/doesn't need a prediction made on any given day
saveRDS(activity_index,paste("activity_index.RDS",sep=""))
########################################
#######################################
##this is a vector to be filled with any instances where a person does not have valid diary data
diary_vector=c()

##these dataframes will be filled by running the loop below (reference in the 'source' command)

keep_id_df_w_lux=data.frame(ID="1",REAL_DATE=as.Date("2021-01-01",format="%Y-%m-%d"))

keep_id_df_wo_lux=data.frame(ID="1",REAL_DATE=as.Date("2021-01-01",format="%Y-%m-%d"))

activity_missing_df=tibble(data.frame(ID="1",REAL_DATE_EXT="1",
                                      NUM_RUNS=0,LONGEST_RUN=0))

whitelight_missing_df=tibble(data.frame(ID="1",REAL_DATE_EXT="1",
                                        NUM_RUNS=0,LONGEST_RUN=0))



#####################################################
#####################################################

source("L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/github/raw_data_loop_processing_single_files_7_14_2022.R")


keep_id_df_w_lux=keep_id_df_w_lux %>% filter(ID!="1")
keep_id_df_wo_lux=keep_id_df_wo_lux %>% filter(ID!="1")

#an inventory of what ids/days had valid lux data and needed a prediction made
saveRDS(keep_id_df_w_lux,"keep_id_df_w_lux.RDS")

#an inventory of what ids/days did NOT have valid lux data and needed a prediction made
saveRDS(keep_id_df_wo_lux,"keep_id_df_without_lux.RDS")

## data frames with info regarding the 'length of missingness' so you can understand how/if imputation was applied to any missing lux data

saveRDS(whitelight_missing_df %>% filter(ID!="1"),"whitelight_missing_df.RDS")

saveRDS(activity_missing_df %>% filter(ID!="1"),"activity_missing_df.RDS")

################################################################################
##next step is to convert rds to pkl



###you can find this at L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/RDS_to_pickle_script_10_25.py
#####make sure to read comments because you only need to run the bottom half of the script

##After this you would run the script that generates the predictions
####you can find this script at L:/SWAN/data analysis/Analyst Foldeojects/actigraphy/Colvin_5_21_2021/scripts/generating_predictions_11_10_2021.py
#####################################################################rs/JF/pr####
################################################################################


###to look at the characteristics of the predictions and how the 'corrected data' compares to the original data look at the following script:
#####L:/SWAN/data analysis/Analyst Folders/JF/projects/actigraphy/Colvin_5_21_2021/scripts/prediction_interval_finish_02_4_2021.R



