#!/usr/bin/env Rscript

source("~/xx_path/scripts/mda_pen.r")

library(readr)
library(tibble)
library(dplyr)
library(mda)
library(caret)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Ensure correct number of arguments
if (length(args) < 3) {
  stop("Usage: script.R <param_file_path> <index> [<mpath>]")
}

param_file_path <- args[1]
idx <- as.numeric(args[2])
param_file <- read_csv(param_file_path)
mpath <- ifelse(length(args) >= 3, args[3], "")


outfol <- param_file[[idx,'outfolder']]
outfol = paste0(outfol,mpath)

print("--------------------------")
print(outfol)

if (!dir.exists(outfol)){
  print("Folder does not exist. Exiting...")
  stop()
}


### base set up MDA, selected features

processed_train <- read_csv(paste0(outfol,"processed_train.csv"),
                      col_types = cols(best = col_factor(levels = c("1", "2")),
                                       bin_hgs = col_factor(levels = c("0", "1")),
                                       bin_filo = col_factor(levels = c("0","1"))))
processed_test <- read_csv(paste0(outfol,"processed_test.csv"),
                     col_types = cols(best = col_factor(levels = c("1", "2")),
                                      bin_hgs = col_factor(levels = c("0", "1")),
                                      bin_filo = col_factor(levels = c("0","1"))))

feature_names = grep("^feature_", names(processed_train), value = TRUE)
selected_features <- read_csv(paste0(outfol,"selected_features.csv"))
selected_features = unlist(selected_features[1,], use.names = FALSE)[-1]

labMap = c("hgs"=1,"filo"=2,"none"=0)
# names(labMap) = c("1","2","0")

### fit
set.seed(1111)
cvPrecision = c()
modelLs = c("best","bin_hgs","bin_filo")

for (perfLab in modelLs){
	print(perfLab)
	# formula_ = as.formula(paste("best ~", paste(feature_names[1:30], collapse = " + "))) ## problem starts at feat 30
	formula_ = as.formula(paste(perfLab,"~", paste(selected_features, collapse = " + ")))
	mod = cv.mda(formula_, data = processed_train, k=5, msub=5)
	assign(paste0(perfLab,".mda"), mod$model)
	cvPrecision = c(cvPrecision, mod$cvPrecision)
}
names(cvPrecision) = modelLs

## defaults for selector
noneBreak = processed_train %>% select(starts_with("bin_")) %>%
mutate(across(everything(), as.character)) %>% mutate(across(everything(), as.numeric)) %>%
colMeans() %>% which.max() %>% names() %>%
sub("^bin_","",.)

tieBreak = cvPrecision[-1] %>% which.max() %>% names() %>% sub("^bin_","",.)


### projection & prediction
Z = rbind(
  predict(best.mda, processed_train, type = "var", dimension=2),
  predict(best.mda, processed_test, type = "var", dimension=2)
)
Z = tibble(
  proj = rep("MDA",nrow(processed_train)+nrow(processed_test)),
  instances = c(processed_train$instances,processed_test$instances),
Z1 = Z[,1], Z2 = Z[,2],
	hgs = c(predict(bin_hgs.mda, processed_train, type="class", dimension=2),
		  predict(bin_hgs.mda, processed_test, type="class", dimension=2)),
	filo = c(predict(bin_filo.mda, processed_train, type="class", dimension=2),
		   predict(bin_filo.mda, processed_test, type="class", dimension=2)),
	bestM = c(predict(best.mda, processed_train, type="class", dimension=2),
		    predict(best.mda, processed_test, type="class", dimension=2)),
	group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test)))
)

Z = bind_cols(Z, selector(Z[,c("hgs","filo")],labMap,tieBreak,noneBreak))
Z = Z %>% mutate(across(where(is.factor), ~ as.numeric(as.character(.))))


### write to csv
write_csv(Z, paste0(outfol,"mda_proj.csv"))
save(best.mda,bin_filo.mda,bin_hgs.mda,Z,noneBreak,tieBreak,cvPrecision,
file = paste0(outfol,"mda_proj.Rdata"))
rm(list = setdiff(ls(), c("param_file","idx","selector","cv.mda","modded")))


#################
### Full Data #####
if (mpath==''){
  
    processed_full <- read_csv(paste0(outfol,"processed_full.csv"), 
                               col_types = cols(best = col_factor(levels = c("1", "2")), 
                                                bin_hgs = col_factor(levels = c("0", "1")), 
                                                bin_filo = col_factor(levels = c("0","1"))))
    feature_names = grep("^feature_", names(processed_full), value = TRUE)
    selected_features <- read_csv(paste0(outfol,"selected_features.csv"))
    selected_features = unlist(selected_features[1,], use.names = FALSE)[-1]
    
    labMap = c("hgs"=1,"filo"=2,"none"=0)
    
    ### fit
    set.seed(1111)
    cvPrecision = c()
    modelLs = c("best","bin_hgs","bin_filo")
    
    for (perfLab in modelLs){
      print(perfLab)
      # formula_ = as.formula(paste("best ~", paste(feature_names[1:30], collapse = " + "))) ## problem starts at feat 30
      formula_ = as.formula(paste(perfLab,"~", paste(selected_features, collapse = " + ")))
      mod = cv.mda(formula_, data = processed_full, k=5, msub=5)
      assign(paste0(perfLab,".mda"), mod$model)
      cvPrecision = c(cvPrecision, mod$cvPrecision)
    }
    names(cvPrecision) = modelLs
    
    ## defaults for selector
    noneBreak = processed_full %>% select(starts_with("bin_")) %>%
      mutate(across(everything(), as.character)) %>% mutate(across(everything(), as.numeric)) %>%
      colMeans() %>% which.max() %>% names() %>% 
      sub("^bin_","",.)
    
    tieBreak = cvPrecision[-1] %>% which.max() %>% names() %>% sub("^bin_","",.)
    
    ### projection & prediction
    Z = predict(best.mda, processed_full, type = "var", dimension=2)
    
    Z = tibble(
      proj = rep("MDA",nrow(processed_full)),
      instances = processed_full$instances,
      Z1 = Z[,1], Z2 = Z[,2],
      hgs = predict(bin_hgs.mda, processed_full, type="class", dimension=2), 
      filo = predict(bin_filo.mda, processed_full, type="class", dimension=2), 
      bestM = predict(best.mda, processed_full, type="class", dimension=2)
    )
    
    Z = bind_cols(Z, selector(Z[,c("hgs","filo")],labMap,tieBreak,noneBreak))
    Z = Z %>% mutate(across(where(is.factor), ~ as.numeric(as.character(.))))
    
    
    ### write to csv
    write_csv(Z, paste0(outfol,"mda_proj_full.csv"))
    save(best.mda,bin_filo.mda,bin_hgs.mda,Z,noneBreak,tieBreak,cvPrecision,
         file = paste0(outfol,"mda_proj_full.Rdata"))
    rm(list = setdiff(ls(), c("param_file","idx","selector","cv.mda")))
  }


