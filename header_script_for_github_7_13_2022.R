


header_fct=function(id_num){

filename=paste(folder_loc,files[id_num],sep="")


header<-read.csv(filename,header=FALSE,nrows=10)

#find the location of the id in the filename and get the id
id_location=max(data.frame(str_locate_all(filename,"/")[[1]])$end)

id=substr(filename,id_location+1,(id_location+7))

diary_status=length(diary_dates %>% 
                      filter(ID==id) %>%
                      pull(ID))

header_ref_df=tibble(data.frame(PHRASE=c("Epoch",
                                         "ActiLife",
                                         "Start Time",
                                         "Start Date")))


search_fct=function(j){which(sapply(seq(from=1,to=length(header$V1)),
                                    function(i,k){str_detect(header$V1[i],as.character(j))}))}


header_ref_df=header_ref_df %>% mutate(REF=as.numeric(modify(.x=PHRASE,.f=search_fct)))

num_criteria <- "[[:digit:]]+"

epoch_parts=str_extract_all(header$V1[header_ref_df$REF[1]],
                            num_criteria)[[1]]

epoch=hms(paste(epoch_parts,collapse=":"))

version_check=str_detect(header$V1[header_ref_df$REF[2]],"v1.6.0")

tibble(data.frame(ID=id,
                      START_TIME=str_remove_all(header$V1[header_ref_df$REF[3]],
                                                letter_space),
                      START_DATE=str_remove_all(header$V1[header_ref_df$REF[4]],
                                                letter_space),
                  VERSION_1_6_0=version_check)) %>%
  mutate( START_DT=as.POSIXct(strptime(paste(START_DATE,START_TIME,sep=" "),
                                       format="%m/%d/%Y %H:%M:%S"),
                              tz="UTC"),
          EPOCH=epoch,
          EPOCH_CUM=as.numeric(EPOCH)/60 )
}

##import time-related data from the header of the file and then 
###create a TimeStamp (REAL_TIME) and DateStamp (REAL_DATE)

