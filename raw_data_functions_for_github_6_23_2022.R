
##choice process, for step 1 of the documentation

choi_process=function(id_num,
                      prefilter) {
  
  ###this section determines the ID number from the filename
  ####this is highly specific to our own data and you will likely want to replace this with 
  ####something that loads in your own data individually
  
  ####to help we have included an example of how the raw data is formatted in our main directory
  
  filename=paste(folder_loc,files[id_num],sep="")
  
  id_location=max(data.frame(str_locate_all(filename,"/")[[1]])$end)
  
  id=substr(filename,id_location+1,(id_location+7))
  
  
  dat=tibble(read.csv(filename,skip=10,colClasses=c(rep("integer",3),rep("NULL",6)))) %>%
    mutate(ID=id) %>%
    left_join(y=header_dat,
              by="ID") 
  
  
  
  dat=dat %>%
    mutate(EPOCH_CUM=cumsum(EPOCH_CUM)-1,
           TimeStamp=START_DT + dminutes(EPOCH_CUM),
           REAL_DATE=date(TimeStamp),
           VM=sqrt((Axis1)^2 + (Axis2)^2 + (Axis3)^2))  
  
  diary_check_num=nrow(dat %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=STOPMON))
  
  dat=dat %>% mutate(DIARY_RANGE_CHECK=if_else(diary_check_num>0,'GOOD','NEEDS_REVIEW'))
  
  colnames(dat)=sapply(colnames(dat),function(i){str_replace_all(i,"\\.","_")})
  
  
  
  ##create a subset of the data for choi processing 
  
  if(prefilter==TRUE) {
    dat=data.frame(dat  %>% filter(REAL_DATE>=STARTMON & REAL_DATE<=STOPMON))
  }
  else {
    dat=data.frame(dat)
  }
  
  
  
  ##process the data using the choi algorithm
  
  dat=tibble(wearingMarking(dataset=dat, 
                            frame=90,
                            perMinuteCts=1,
                            TS="TimeStamp",
                            cts="VM",
                            allowanceFrame=2,
                            newcolname="wearing")) 
  
  dat=dat %>%group_by(REAL_DATE) %>%
    mutate(MIN_WEARING=min(TimeStamp[wearing=="w"]),
           MAX_WEARING=max(TimeStamp[wearing=="w"])) %>%
    ungroup()
  

  ##this is the number of activity minutes per day
  obs_count=dat %>% group_by(days) %>%
    summarize(OBSMIN=length(unique(TimeStamp[wearing=="w"])),
              .groups="keep")
  
  
  
  ############creating discretized variables
  
  ##MVAR VARIABLES
  # mlight<-ifelse(100<=ctd & ctd<760,1,0)
  # mmodlife<-ifelse(760<=ctd & ctd<1952,1,0)
  # mmodwalk<-ifelse(1952<=ctd & ctd<5725,1,0)
  # mvigorous<-ifelse(5725<=ctd,1,0)
  
  mvar=dat %>% 
    filter(wearing=="w") %>%
    select(days, TimeStamp, Axis1) %>%
    mutate(MVAR= cut(Axis1, 
                     breaks=c(100,760,1952,5725,10000000000000000),
                     labels=c("MLIGHT","MMODLIFE","MMODWALK",
                              "MVIGOROUS"),
                     include.lowest=TRUE,
                     right=FALSE),
           ONE=1
    ) %>% 
    pivot_wider(id_cols=c("days"),
                names_from="MVAR",
                values_from="ONE",
                values_fn=sum,
                values_fill=0) %>%
    select(-`NA`)
  
  ###FREEDSON VARIABLES
  # flight<-ifelse(100<=ctd & ctd<1952,1,0)
  # fmoderate<-ifelse(1952<=ctd & ctd<5725,1,0)
  # fvigorous<-ifelse(5725<=ctd,1,0)
  
  freedson=dat %>%
    filter(wearing=="w") %>%
    select(days,TimeStamp,Axis1) %>%
    mutate(FREE=cut(Axis1,
                    breaks=c(100,1952,5725,100000000000000),
                    labels=c("FLIGHT","FMODERATE","FVIGOROUS"),
                    include.lowest=TRUE,
                    right=FALSE),
           ONE=1) %>% 
    pivot_wider(id_cols="days",
                names_from="FREE",
                values_from="ONE",
                values_fn=sum,
                values_fill=0) %>% 
    select(-`NA`) 
  
  ####NHAINES VARIABLES
  # nlight<-ifelse(100<=ctd & ctd<2020,1,0)
  # nmoderate<-ifelse(2020<=ctd & ctd<5999,1,0)
  # nvigorous<-ifelse(5999<=ctd,1,0)
  
  nhaines=dat %>% 
    filter(wearing=="w") %>%
    select(days,TimeStamp,Axis1) %>%
    mutate(NHAINE=cut(Axis1,
                      breaks=c(100,2020,5999,100000000000000),
                      labels=c("NLIGHT","NMODERATE","NVIGOROUS"),
                      include.lowest=TRUE,
                      right=FALSE),
           ONE=1) %>% 
    pivot_wider(id_cols="days",
                names_from="NHAINE",
                values_from="ONE",
                values_fn=sum,
                values_fill=0) %>% 
    select(-`NA`) 
  
  ##VECTOR MAGNITUDE VARIABLES
  #   vmsedentary<-ifelse(0<=VM & VM<76,1,0)
  # vmllight<-ifelse(76<=VM & VM<903,1,0)
  # vmhlight<-ifelse(903<=VM & VM<2075,1,0)
  # vmmvpa<-ifelse(2075<=VM,1,0)
  
  vecm=  dat %>% 
    filter(wearing=="w") %>%
    select(days,TimeStamp,VM) %>%
    mutate(FREE=cut(VM,
                    breaks=c(0,76,903,2075,100000000000000),
                    labels=c("VMSEDENTARY","VMLLIGHT","VMHLIGHT","VMMVPA"),
                    include.lowest=TRUE,
                    right=FALSE),
           ONE=1) %>% 
    pivot_wider(id_cols="days",
                names_from="FREE",
                values_from="ONE",
                values_fn=sum,
                values_fill=0) 
  
  ### ALL OF THE OVERLAPPING VARIABLES
  
  # fmvpa<-ifelse(fmoderate==1 | fvigorous==1,1,0)   
  # mmvpa1<-ifelse(mmodlife==1 | mmodwalk==1 | mvigorous==1,1,0)
  # mmvpa2<-ifelse(mmodwalk==1 | mvigorous==1,1,0)
  # mmoderate<-ifelse(760<=ctd & ctd<=5725,1,0)
  # nmvpa<-ifelse(nmoderate==1 | nvigorous==1,1,0)
  # mets_f1=1.439008 + (0.000795 * ctd)
  # mets_s1=2.606 + (0.0006863 * ctd)
  
  overlapping_vars= dat %>% 
    filter(wearing=="w") %>%
    select(days,TimeStamp,Axis1) %>%
    mutate(MMODERATE=ifelse(Axis1>=760 & Axis1<5725,1,0),
           MMVPA1=ifelse(Axis1>=760,1,0),
           MMVPA2=ifelse(Axis1>=1952,1,0),
           FMVPA= ifelse(Axis1>=1952,1,0),
           NMVPA=ifelse(Axis1>=2020,1,0),
           METS_F1=((Axis1*0.000795) +1.439008),
           METS_S1=((Axis1*0.0006863) +2.606),
           INACTIVE=ifelse(Axis1<100,1,0),
           ACTIVE=ifelse(Axis1>=100,1,0))%>% 
    pivot_longer(cols=c("MMODERATE","MMVPA1","MMVPA2","FMVPA","NMVPA",
                        "METS_F1","METS_S1","INACTIVE","ACTIVE"))%>%
    group_by(days,name) %>% 
    summarize(SUM=sum(value,na.rm=TRUE),
              .groups="keep") %>%
    ungroup() %>%
    pivot_wider(id_cols="days",
                names_from="name",
                values_from="SUM")
  
  
  
  
  
  dat_final=distinct(dat %>% select(ID,REAL_DATE,days,START_DATE,
                                    START_DT,VERSION_1_6_0,EPOCH, STARTWAT,
                                    STOPWAT,STARTMON,STOPMON,TRNDT, 
                                    REAL_DATE,DIARY_RANGE_CHECK,
                                    MIN_WEARING,MAX_WEARING,
                                    weekday)) %>%
    left_join(obs_count,by="days") %>% 
    left_join(overlapping_vars,by="days") %>% 
    left_join(freedson,by="days")%>% 
    left_join(vecm,by="days")%>% 
    left_join(nhaines,by="days")%>% 
    left_join(mvar,by="days") 
  
  disc_vars_to_have=c("MMODERATE","MMVPA1","MMVPA2","FMVPA","NMVPA",
                      "VMSEDENTARY","VMLLIGHT","VMHLIGHT","VMMVPA",
                      "NLIGHT","NMODERATE","NVIGOROUS",
                      "FLIGHT","FMODERATE","FVIGOROUS",
                      "MLIGHT","MMODLIFE","MMODWALK","MVIGOROUS",
                      "ACTIVE","INACTIVE")
  
  
  missing_disc_vars=setdiff(disc_vars_to_have,
                            colnames(dat_final))
  
  
  dat_final=data.table(dat_final)
  
  dat_final[,c(missing_disc_vars):=0]
  
  dat_final=tibble(dat_final)
  
  dat_final %>% relocate(c("ID", 
                           "days",
                           "weekday",
                           "REAL_DATE",
                           "START_DATE",
                           "START_DT",
                           "EPOCH",
                           "STARTMON","STOPMON",
                           "STARTWAT","STOPWAT",
                           "TRNDT",
                           "VERSION_1_6_0",
                           "OBSMIN",
                           "DIARY_RANGE_CHECK",
                           "MIN_WEARING",
                           "MAX_WEARING",
                           all_of(c(disc_vars_to_have))))
}


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################





################################################################################
################################################################################

##this function extends the 'search area' for each day so that it is not limited to a strict 24-hour grid

##this is done for preparing the data to be input into the model

date_extend_fct=function(i,df){  
  datt=df %>% left_join(activity_index %>% 
                          filter(REAL_DATE==i) %>% 
                          select(ID,MIN_TS,MAX_TS),
                        by=c("ID"))
  
  datt=datt %>% filter(TimeStamp>=MIN_TS & TimeStamp<=MAX_TS) %>% 
    mutate(REAL_DATE_EXT=as.character(i),
           VM=sqrt((Axis1^2)+(Axis2^2)+(Axis3^2)))
  
  excl_test=nrow(dat %>% filter(SWStatus =='EXCLUDED'))
  
  datt=tibble(wearingMarking(dataset=data.frame(datt), 
                             frame=90,
                             perMinuteCts=1,
                             TS="TimeStamp",
                             cts="VM",
                             allowanceFrame=2,
                             newcolname="wearing"))
  
  return(datt)}
















