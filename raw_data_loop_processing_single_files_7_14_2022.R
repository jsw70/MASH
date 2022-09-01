
##this is the loop for processing the raw actigraphy/sleep/lux data and putting into the apropriate folders
##depending on whether or not the lux data is present/useable.

for (i in 1:length(files)){
  
  
  filename=paste(folder_loc,files[i],sep="")
  
  id_location=max(data.frame(str_locate_all(filename,"/")[[1]])$end)
  
  iddd=substr(filename,id_location+1,(id_location+7))
  
  ##if they have data to be predicted
  if (length(intersect(iddd,unique((activity_index %>% filter(PRED_KEEP==1))$ID)))>0){
    
    #get lux filename (this is used for next if test)
    lux_filename=c(files_lux)[str_detect(files_lux,iddd)]
    
    
    
    dat=tibble(read.csv(filename,skip=10,colClasses=c(rep("integer",9)))) %>%
      mutate(ID=iddd) %>%
      left_join(y=header_dat,
                by="ID") %>%
      mutate(START_DATE=as.Date(START_DATE,format="%m/%d/%Y"))
    
    dat=dat%>%
      mutate(EPOCH_CUM=cumsum(EPOCH_CUM)-1,
             TimeStamp=as.POSIXct(as.character((START_DT + dminutes(EPOCH_CUM))),tz="GMT"),
             REAL_DATE=date(TimeStamp))
    
    dat=dat %>% rename_at(colnames(dat),
                          ~str_replace_all(.x,'[.]',''))
    
    dat=dat %>% select(ID,TimeStamp,Axis1,Axis2,Axis3,REAL_DATE,STARTMON,STOPMON) %>% 
      relocate(ID)
    
    #############################################################################
    # if they have lux data available
    if (length(lux_filename)>0) {   
      header_check=readLines(paste(folder_loc_lux,
                                   lux_filename,
                                   sep=""),n=40)
      
      lux_data_begin=max(setdiff(seq(from=1,to=40), seq(from=1,to=40)[str_detect(header_check,"")]))
      
      dat_lux=tibble(read.csv(paste(folder_loc_lux,
                                    lux_filename,
                                    sep=""),skip=lux_data_begin-2))
      
      dat_lux =dat_lux%>%
        rename_at(colnames(dat_lux),
                  ~str_replace_all(.x,'[.]','')) %>% 
        mutate(ID=iddd,
               Date=as.Date(Date,format="%m/%d/%Y"),
               TimeStamp=as_datetime(paste(Date,Time, sep=" ")))%>% 
        select(ID,Activity,WhiteLight,SWStatus,TimeStamp) %>%
        relocate(ID)
      
      dat_lux=dat_lux %>% mutate(across(c("WhiteLight","Activity"),
                                        ~ifelse(.x=='.',0,.x)))
      
      ##########################################################
      
      ##finding the dates neede for prediction
      realdts=sort(unique((activity_index %>% filter(ID==iddd & PRED_KEEP==1))$REAL_DATE))
      
      dat=dat %>% left_join(dat_lux,
                            by=c("ID","TimeStamp"))
      
      ##checking to see if they have diary data
      diary_check_num=nrow(dat %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=STOPMON))
      
      #if they have diary data filter it so that we are only looking at days when monitor was worn
      if (diary_check_num>0) {
        dat=dat  %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=(STOPMON+1)) %>%
          select(-STARTMON,-STOPMON)
      }
      
      ##if they don't have it dump it inot the diary_vector_
      else {diary_vector<<-c(diary_vector,id)}
      
      
      ##create these lists to be filled by respective REAL_DATE_EXT's that apply
      w_lux_list=list()
      wo_lux_list=list()
      
      
      for (i in realdts) {
        
        ##this exists solely for checking code when errors are thrown
        check=i
        
        ##add the MIN_TS AND MAX_TS
        ##they will fall outside of the actual date boundary so they need to be set first and then we 
        ###subset everything with these varaibles
        datt=dat %>% left_join(activity_index %>% 
                                 filter(REAL_DATE==i) %>% 
                                 select(ID,MIN_TS,MAX_TS),
                               by=c("ID"))
        
        datt=datt %>% filter(TimeStamp>=MIN_TS & TimeStamp<=MAX_TS) %>% 
          mutate(REAL_DATE_EXT=as.character(i),
                 VM=sqrt((Axis1^2)+(Axis2^2)+(Axis3^2)))
        
        datt=datt %>% mutate(across(c("Activity","WhiteLight"),
                                    ~ifelse(is.na(.x)==TRUE,1,0),
                                    .names="{col}_MISSING"))
        
        ##finding the length of missingness of lux data
        
        ######################################
        ff=rle(datt$Activity_MISSING)
        
        act_missing_temp=data.frame(ID=unique(datt$ID),
                                    REAL_DATE_EXT=as.character(i),
                                    NUM_RUNS=sum(c(ff$values)),
                                    LONGEST_RUN=max(c(ff$lengths)*c(ff$values)))
        
        
        
        gg=rle(datt$WhiteLight_MISSING)     
        
        wl_missing_temp=data.frame(ID=unique(datt$ID),
                                   REAL_DATE_EXT=as.character(i),
                                   NUM_RUNS=sum(c(gg$values)),
                                   LONGEST_RUN=max(c(gg$lengths)*c(gg$values)))
        
        ######################################
        
        ##this is creating test criteria for exclusion of lux data
        excl_test1=nrow(datt %>% filter((SWStatus =='EXCLUDED') ))
        
        excl_test2=suppressWarnings(max((wl_missing_temp$LONGEST_RUN), (act_missing_temp$LONGEST_RUN)))
        
        ##if the SWStatus =excluded for any records or if more than 10 whitelight/acitivty records are missing in a row
        if (excl_test1>0 | excl_test2>=10) {
          
          
          datt=tibble(wearingMarking(dataset=data.frame(datt %>% select(-WhiteLight,
                                                                        -Activity,
                                                                        -SWStatus,
                                                                        
                                                                        -Activity_MISSING,
                                                                        -WhiteLight_MISSING)), 
                                     frame=90,
                                     perMinuteCts=1,
                                     TS="TimeStamp",
                                     cts="VM",
                                     allowanceFrame=2,
                                     newcolname="wearing")) 
          
          
          wo_lux_list[[i]]=datt
        }
        else if(excl_test1==0 & excl_test2<=10){
          
          activity_missing_df<<-rbind(activity_missing_df,
                                      act_missing_temp)
          
          whitelight_missing_df<<-rbind(whitelight_missing_df,
                                        wl_missing_temp)
          
          datt=datt %>% mutate(across(c("Activity","WhiteLight"),~na_ma(as.numeric(.x),k=5,weighting="simple"))) %>%
            select(-Activity_MISSING,-WhiteLight_MISSING)
          
          
          datt=tibble(wearingMarking(dataset=data.frame(datt), 
                                     frame=90,
                                     perMinuteCts=1,
                                     TS="TimeStamp",
                                     cts="VM",
                                     allowanceFrame=2,
                                     newcolname="wearing")) 
          
          
          
          
          w_lux_list[[i]]=datt
        }
      }
      
      ## here we have finished the 'for i in realdts' in cases where lux data was at least 
      ###initially present (although it might not have made it to the end depending on missingness/excluded status)
      
      w_lux_df=bind_rows(w_lux_list)
      
      
      
      wo_lux_df=bind_rows(wo_lux_list)
      
      
      
      if (nrow(w_lux_df)>0) {      
        keep_id_df_w_lux=rbind(keep_id_df_w_lux,
                               w_lux_df %>% select(ID,REAL_DATE) %>% distinct())
        saveRDS(w_lux_df,paste('with_lux/',iddd,'.RDS',sep=""))
      }
      
      if (nrow(wo_lux_df)>0) { 
        keep_id_df_wo_lux=rbind(keep_id_df_wo_lux,
                                wo_lux_df %>% select(ID,REAL_DATE) %>% distinct())
        saveRDS(wo_lux_df,paste('without_lux/',iddd,'.RDS',sep=""))
      }
      
      
    }
    ##here we end the 'if no lux data==false' so this is for non-lux data
    else if (length(lux_filename)==0) {
      
      
      diary_check_num=nrow(dat %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=STOPMON))
      
      if (diary_check_num>0) {
        dat=dat  %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=(STOPMON+1)) %>%
          select(-STARTMON,-STOPMON)}
      
      else {diary_vector<<-c(diary_vector,id)}
      
      realdts=sort(unique((activity_index %>% filter(ID==iddd & PRED_KEEP==1))$REAL_DATE))
      
      wo_lux_list=list()
      
      
      for (i in realdts) {
        
        dattt=dat %>% left_join(activity_index %>% 
                                  filter(REAL_DATE==i) %>% 
                                  select(ID,MIN_TS,MAX_TS),
                                by=c("ID"))
        
        dattt=dattt %>% filter(TimeStamp>=MIN_TS & TimeStamp<=MAX_TS) %>% 
          mutate(REAL_DATE_EXT=as.character(i),
                 VM=sqrt((Axis1^2)+(Axis2^2)+(Axis3^2)))
        
        
        dattt=tibble(wearingMarking(dataset=data.frame(dattt), 
                                    frame=90,
                                    perMinuteCts=1,
                                    TS="TimeStamp",
                                    cts="VM",
                                    allowanceFrame=2,
                                    newcolname="wearing"))
        
        
        wo_lux_list[[i]]=dattt
        
      }
      
      wo_lux_df=bind_rows(wo_lux_list)
      
      if (nrow(wo_lux_df)>0) { 
        keep_id_df_wo_lux=rbind(keep_id_df_wo_lux,
                                wo_lux_df %>% select(ID,REAL_DATE) %>% distinct())
        saveRDS(wo_lux_df,paste('without_lux/',iddd,'.RDS',sep=""))
      }
      
      
      
    } 
    #ending if there is no lux data
  }
  #ending if there there is no data to be predicted
  else {print("id not for prediction")}
  
}
##ending the loop!
