library(tidyverse)
library(haven)
library(stringr)
library(lubridate)


#load processed sleep data
##refer to template for guidance on how file is prepared
sleep_day=tibble(read_sas(paste(working_dir,'actslpall151802.sas7bdat',sep="")))

sleep_day=sleep_day %>% filter(USECD15=='G' & SLPTYPE15=='Primary')

sleep_day=sleep_day %>% rename_at(colnames(sleep_day),
                                    ~toupper(.x))

##this loads in previously created/cleaned activity data
active_day=tibble(readRDS(paste(working_dir,"raw_data_processed_",current_date,'/',activity_data_name,sep="")))



active_day=active_day %>% rename_at(colnames(active_day),
                                    ~toupper(.x))

active_day=active_day %>% filter(OBSMIN>=600 & 
                        is.na(VMSEDENTARY)==FALSE & 
                        REAL_DATE>=STARTMON &
                        REAL_DATE<=STOPMON) %>%
  mutate(STRDATE_ALT=as_datetime(REAL_DATE,tz="UTC"))

active_day %>% summarize(RECORD_COUNT=n(),
                         ID_COUNT=length(unique(ID)))

sleep_day %>% summarize(RECORD_COUNT=n(),
                         ID_COUNT=length(unique(ID)))

active_day=active_day %>% group_by(ID) %>%
  mutate(DAY_COUNT=n()) %>%
  ungroup()%>% filter(DAY_COUNT>=4)

#create reliable days variable to determine consecutiveness
active_day=active_day %>%
  group_by(ID) %>%
  mutate(MIN_STRDATE_ALT=min(STRDATE_ALT,na.rm=TRUE),
         DAY_CALC=round((as.numeric(difftime(STRDATE_ALT,
                                            MIN_STRDATE_ALT,
                                            units='mins'))/1440),digits=0)+1)



active_day %>% mutate(IN_SLEEP=ifelse(ID %in% unique(sleep_day$ID),1,0)) %>%
  group_by(IN_SLEEP) %>% 
  summarize(RECORD_COUNT=n(),
            ID_COUNT=length(unique(ID)),
            .groups="keep")

sleep_day %>% mutate(IN_ACTIVITY=ifelse(ID %in% unique(active_day$ID),1,0)) %>%
  group_by(IN_ACTIVITY)%>% 
  summarize(RECORD_COUNT=n(),
            ID_COUNT=length(unique(ID)),
            .groups="keep")

match=sleep_day %>% inner_join(active_day,
                         by="ID") %>% 
select(ID,DAY_CALC,REAL_DATE,STRDATE_ALT,
       NIGTNUM15,SLPITSRT15,SLPITEND15,ACSLPTOT15,
       VMSEDENTARY)


#check to make sure numbers match with numbers from ealier sql tables;
# match %>% summarize(RECORD_COUNT=n(),
#                         ID_COUNT=length(unique(ID)))

#calculate metrics showing distance between days
combined=match %>% 
  mutate(DATE_DIFF=as.numeric(difftime(SLPITSRT15,STRDATE_ALT,units='mins')),
         ADD_MIN=abs(DATE_DIFF-1440)) %>%
  group_by(ID,DAY_CALC) %>%
  mutate(MIN_ADD_MIN=min(ADD_MIN,na.rm=TRUE)) %>%
  filter(ADD_MIN==MIN_ADD_MIN) %>%
  select(-MIN_ADD_MIN) %>%
  ungroup()


combined %>% mutate(SLEEP_DAYS=paste(ID,SLPITSRT15,sep="_")) %>%
  summarize(sleep_days=length(unique(SLEEP_DAYS)),
            total_days=n(),
            total_ID=length(unique(ID)))


##look at people who have null values for nigtnum;
##it seems like all of them have legit sleep start and end data;

combined %>% filter(ID %in% c(combined %>% 
                                filter(is.na(NIGTNUM15)==TRUE) %>% 
                                pull(ID)))

##now we need to get the 'min' of ADD_MIN for cases where there are duplicate SLPITSRT15 values
##also get the max DAY_CALC per person
combined=combined %>% 
  group_by(ID,SLPITSRT15) %>%
  mutate(ADD_MIN_MIN=min(ADD_MIN,na.rm=TRUE)) %>%
  ungroup()%>%
  group_by(ID) %>%
  mutate(DAY_CALC_MAX=max(DAY_CALC,na.rm=TRUE),
         SLPSRT_COUNT=length(unique(SLPITSRT15[is.na(SLPITSRT15)==FALSE])),
         ADD_MIN_LAG=lag(ADD_MIN)) %>%
  ungroup()

combined %>% summarize(RECORD_COUNT=n(),
                    ID_COUNT=length(unique(ID)))

combined %>% filter(is.na(ACSLPTOT15)==FALSE &
                      is.na(VMSEDENTARY)==FALSE) %>%
  summarize(RECORD_COUNT=n(),
                       ID_COUNT=length(unique(ID)))



combined=combined %>% mutate(
  NEW_DAY_CALC=ifelse(
    # if the activity day and sleep-day datetime is not missing and the distance between the start of sleep and the 
    # begining of the day is the smallest possible (being under 1440); 
    (is.na(DAY_CALC)==FALSE & is.na(SLPITSRT15)==FALSE &
                        ADD_MIN_MIN==ADD_MIN) 
    |
      # *if it is the last day and there is no sleep day to join to it;     ,
      (DAY_CALC==DAY_CALC_MAX & 
       ADD_MIN != ADD_MIN_MIN & 
        ADD_MIN_MIN==ADD_MIN_LAG)
  |
    #sleep time is missing but not the activity time
    (is.na(SLPITSRT15)==TRUE & is.na(DAY_CALC)==FALSE),
                        DAY_CALC,
                        ifelse(
   # if nothing is missing but the distance between day and night is not minimized;
    (is.na(DAY_CALC)==FALSE & 
       is.na(SLPITSRT15)==FALSE & 
       ADD_MIN_MIN != ADD_MIN) |
  #if the activity day and sleep-day datetime is not missing and the distance between the start of sleep and the 
  #begining of the day is the smallest possible (being under 1440)
      (is.na(DAY_CALC)==TRUE &
         is.na(SLPITSRT15)==TRUE),
         NA,
  10000))
)

combined %>% filter(NEW_DAY_CALC==10000)

combined %>% filter(is.na(NEW_DAY_CALC)==FALSE) %>%
  summarize(RECORD_COUNT=n(),
                       ID_COUNT=length(unique(ID)))


#**here we want to preserve a record for the eventual lag variable but don't want to use the total sleep varible
#found in that record line, thus creating the disregard sleep variable;

#**this gets rid of sleep start times for two potential reasons
#1)There is a missing last sleep day and the previous sleep day has been duplicated
#2) There is an activity day missing where there is not a sleep day missing so a 'previous' sleep day is being paired with the wrong activity day;

combined=combined %>% group_by(ID) %>%
  mutate(NEW_DAY_CALC_MAX=max(NEW_DAY_CALC,na.rm=TRUE)) %>%
  ungroup()%>%
  mutate(DISREGARD_SLEEP=ifelse(
    (NEW_DAY_CALC==NEW_DAY_CALC_MAX & ADD_MIN_MIN != ADD_MIN) |
      (is.na(NEW_DAY_CALC)==TRUE) |
      (is.na(NEW_DAY_CALC)==FALSE & ADD_MIN>1440),
    1,0))

combined %>% group_by(DISREGARD_SLEEP) %>%
  summarize(RECORD_COUNT=n(),
            ID_COUNT=length(unique(ID)),
            .groups="keep")
  

#**CREATE THE LAGGED SLEEP END (SLPITEND15) VARAIBLE (BEING THE WAKE UP TIME), DO THIS THROUGH MERGING, NOT JUST USING LAG;

lagged_sleep=combined %>% select(ID,NEW_DAY_CALC,PREV_END=SLPITEND15,ACSLPTOT15,PREV_SRT=SLPITSRT15) %>%
  mutate(NEW_DAY_CALC=NEW_DAY_CALC+1) %>%
  filter(is.na(NEW_DAY_CALC)==FALSE)


combined_days_final=combined %>% left_join(lagged_sleep,
                                      by=c("ID","NEW_DAY_CALC"))

combined_days_final %>% filter(is.na(PREV_END)==FALSE) %>%
  summarize(RECORD_COUNT=n(),
            ID_COUNT=length(unique(ID)))

##NOW CREATE FUTURE SLEEP!

future_sleep=combined %>% select(ID,NEW_DAY_CALC,FUTURE_END=SLPITEND15,FUTURE_SRT=SLPITSRT15) %>%
  mutate(NEW_DAY_CALC=NEW_DAY_CALC-1) %>%
  filter(is.na(NEW_DAY_CALC)==FALSE)


combined_days_final=combined_days_final %>% left_join(future_sleep,
                                           by=c("ID","NEW_DAY_CALC"))

combined_days_final %>% filter(is.na(FUTURE_SRT)==FALSE) %>%
  summarize(RECORD_COUNT=n(),
            ID_COUNT=length(unique(ID)))


####


combined_days_final=combined_days_final %>% 
  group_by(ID) %>%
  mutate(STARTING=ifelse(DAY_CALC==1,1,0)) %>%
  ungroup()

combined_days_final=combined_days_final %>%
  mutate(INTERVAL_CAT=ifelse(
    is.na(PREV_END)==FALSE & 
      is.na(SLPITSRT15)==FALSE & 
      DISREGARD_SLEEP==0,
    "1",
    ifelse(is.na(PREV_END)==FALSE & 
             ((is.na(SLPITSRT15)==FALSE & DISREGARD_SLEEP==1) |
                (is.na(SLPITSRT15)==TRUE)),
           "2",
           ifelse(is.na(PREV_END)==TRUE & 
                    is.na(SLPITSRT15)==FALSE & 
                    DISREGARD_SLEEP==0,
                  "3",
                  ifelse(is.na(PREV_END)==TRUE & 
                           ((is.na(SLPITSRT15)==TRUE) | 
                              (is.na(SLPITSRT15)==FALSE & DISREGARD_SLEEP==1)),
                         "4","INVESTIGATE")))))

combined_days_final %>% group_by(INTERVAL_CAT,STARTING) %>%
  summarize(COUNT=n(),
            .groups="keep")


combined_days_final =combined_days_final %>%
  mutate(TBS=as.numeric(difftime(SLPITSRT15,STRDATE_ALT,units="mins")),
         TBW=as.numeric(difftime(PREV_END,STRDATE_ALT,units="mins")))

##if people either: 
#1) woke up suspiciously early (waking up before 11pm)
#2) woke up suspcisioualy late (waking up after 4pm)
#3) fell asleep between 12am and 8am

##then we are concerned that the wake-sleep join might not be 
#valid and therefore recategorize these few instances to being INTERVAL_CAT=="4"

combined_days_final = combined_days_final %>%
  mutate(INTERVAL_CAT_ALT=ifelse((INTERVAL_CAT %in% c("1") & (TBS<480| (TBW>960 | TBW< -60))) |
                                    (INTERVAL_CAT %in% c("3") & TBS<480) | 
                                   (INTERVAL_CAT =="2" & (TBW>960 | TBW< -60)),
                                 "4",
                                 INTERVAL_CAT
  ))

early_wakers=combined_days_final %>% filter((INTERVAL_CAT %in% c("1","2") & ( TBW< -60))) %>%
  mutate(PROBLEM_TAG="early_waker")

late_wakers=combined_days_final %>% filter((INTERVAL_CAT %in% c("1","2") & ( TBW > 960))) %>% 
  mutate(PROBLEM_TAG="late_waker")

early_sleepers=combined_days_final %>% filter((INTERVAL_CAT %in% c("1","3") & TBS<480)) %>%
  mutate(PROBLEM_TAG="early_sleeper")


problems=rbind(early_wakers,late_wakers)

problems=rbind(problems,early_sleepers)

##################


saveRDS(combined_days_final,paste('TIER_1_',current_date,'.Rda',sep=""))

write.csv(problems,
        paste('TIER_1_PROBLEMS_',current_date,'.csv',sep=""))

write.csv(combined_days_final,
          paste('TIER_1_',current_date,'.csv',sep=""),
          row.names=FALSE)









