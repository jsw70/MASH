##THIS CREATES THE ACTIVITY INDEX
##this is where the variable 'num_valid_days' is used

activity_index=distinct(activity_data  %>% 
                          filter(REAL_DATE>=STARTMON & REAL_DATE<=(STOPMON+1)) %>%
                          select(ID,REAL_DATE,OBSMIN))


activity_index=activity_index %>% 
  left_join(y=distinct(interval_dat %>% select(ID,REAL_DATE,INTERVAL_CAT=INTERVAL_CAT_ALT,DATE_DIFF)),
            by=c("ID","REAL_DATE"))

activity_index=activity_index %>% filter(OBSMIN>=600) %>%
  group_by(ID) %>%
  mutate(DAY_COUNT=n())%>% ungroup()

activity_index=activity_index %>% filter(DAY_COUNT>=num_valid_days) %>%
  select(-DAY_COUNT)

activity_index=activity_index %>% group_by(ID) %>% 
  mutate(MIN_REAL_DATE=min(REAL_DATE))

activity_index=activity_index %>% 
  mutate(INTERVAL_CAT=ifelse(is.na(INTERVAL_CAT)==TRUE,"4",INTERVAL_CAT),
         INTERVAL_CAT_ALT=ifelse(INTERVAL_CAT=="3" & DATE_DIFF< 0,"4",INTERVAL_CAT),
         STARTING_ALT=ifelse(REAL_DATE==MIN_REAL_DATE & INTERVAL_CAT=="3",1,0),
         PRED_KEEP=ifelse(INTERVAL_CAT !="1" & STARTING_ALT !=1 & OBSMIN>=600,1,0 ))


activity_index %>% group_by(INTERVAL_CAT,STARTING_ALT,APPLY_PREDICTION=PRED_KEEP) %>%
  summarize(COUNT=n(),
            .groups="keep")

activity_counter=activity_index %>% mutate(TAG=paste(ID,REAL_DATE,sep="_"))


activity_counter %>% group_by(PRED_KEEP) %>%
  summarize(COUNT_RECORDS=n(),
            COUNT_ID=length(unique(ID)),
            .groups="keep") %>% ungroup() %>%
  rename(APPLY_PREDICTION=PRED_KEEP)


activity_counter_table=activity_counter %>% 
  group_by(ID) %>%
  mutate(TOTAL_NUM=n(),
         TOTAL_NUM_CORRECTED=length(unique(TAG[PRED_KEEP==1])))%>%
  select(ID,TOTAL_NUM,TOTAL_NUM_CORRECTED,OBSMIN) %>% distinct() %>%
  group_by(TOTAL_NUM,TOTAL_NUM_CORRECTED) %>%
  summarize(COUNT=n(),
            .groups="keep") %>%
  pivot_wider(id_cols="TOTAL_NUM",names_from="TOTAL_NUM_CORRECTED",
              values_from="COUNT",names_prefix="CORRECT_") %>%
  ungroup()

# activity_counter_table%>%
#   mutate(across(colnames(activity_counter_table),
#                 ~ifelse(is.na(.x)==TRUE,0,.x)))


# activity_counter %>% 
#   group_by(ID) %>%
#   mutate(TOTAL_NUM=n(),
#          TOTAL_NUM_CORRECTED=length(unique(ID[PRED_KEEP==1])))%>%
#   group_by(TOTAL_NUM) %>%
#   summarize(COUNT=n(),
#             NUMBER_OF_CORRECTIONS=sum(TOTAL_NUM_CORRECTED),
#             .groups="keep") %>% ungroup() %>%
#   mutate(CUM_COUNT=cumsum(COUNT),
#          CUM_NUMBER_OF_CORRECTIONS=cumsum(NUMBER_OF_CORRECTIONS),
#          PERC_WITHIN_GROUP=NUMBER_OF_CORRECTIONS/COUNT,
#          PERC_OF_WHOLE=CUM_NUMBER_OF_CORRECTIONS/sum(COUNT),
#          CUM_PERC_OF_WHOLE=CUM_NUMBER_OF_CORRECTIONS/CUM_COUNT)



# activity_counter %>% 
#   group_by(ID) %>%
#   mutate(TOTAL_NUM=n(),
#          TOTAL_NUM_CORRECTED=length(unique(TAG[PRED_KEEP==1])))%>%
#   select(ID,TOTAL_NUM,TOTAL_NUM_CORRECTED,OBSMIN) %>% distinct() %>%
#   group_by(TOTAL_NUM,TOTAL_NUM_CORRECTED) %>%
#   summarize(COUNT=n(),
#             .groups="keep")

larger_ids=unique(activity_counter %>% filter(PRED_KEEP==1) %>% pull(ID))


activity_index=activity_index %>% group_by(ID,REAL_DATE) %>%
  mutate(MIN_TS=as_datetime(REAL_DATE)-(300*60),
         MAX_TS=as_datetime(REAL_DATE)+((1440+300)*60)) %>%
  ungroup()

cat_type_count_df=activity_index %>% group_by(INTERVAL_CAT,STARTING_ALT,PRED_KEEP) %>% 
  summarize(COUNT=n(),.groups="keep") %>% ungroup() %>%
  mutate(TOTAL=sum(COUNT),
         PERCENTAGE=COUNT/TOTAL) %>%
  select(-TOTAL)
