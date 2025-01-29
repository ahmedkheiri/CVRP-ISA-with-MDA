library(tibble)
library(dplyr)
library(mda)
library(caret)
library(purrr)

cv.mda <- function(formula, data, k, msub) {
  
  ylab = all.vars(formula)[1]
  nclass = length(levels(data[[ylab]]))
  folds = createFolds(data[[ylab]], k)
  
  # subclasses to tune
  s_list <- setNames(rep(list(1:msub), nclass), paste0(1:nclass))
  subclasses <- do.call(expand.grid, s_list)
  subclasses <- map(pmap(subclasses, c),unname)[-1]
  
  # Results storage
  results <- data.frame(
    subclasses = I(subclasses), # Store list of vectors
    CV_Error = numeric(length(subclasses))
  )
  if (nclass==2){
    precisions = numeric(length(subclasses))
  }
  
  # Loop over subclass combinations
  for (i in seq_along(subclasses)) {
    s <- subclasses[[i]]
    errors <- numeric(k)
    prec_fold <- numeric(k)
    
    for (j in 1:k) {
      # Split into training and testing
      train_data <- data[-folds[[j]], ]
      test_data <- data[folds[[j]], ]
      
      # Train model
      model <- mda(formula, data = train_data, subclasses = s)
      predictions <- predict(model, test_data, type="class")
      
      # Calculate misclassification error
      errors[j] <- mean(predictions != data[[ylab]][folds[[j]]])
      if (nclass==2){
        prec_fold[j] <- precision(predictions, data[[ylab]][folds[[j]]],"1")}
    }
    
    # Store the average error for this subclass combination
    results$CV_Error[i] <- mean(errors)
    if (nclass==2){
      precisions[i] <- mean(prec_fold)}}
  print(results)
  
  bestS = subclasses[[which.min(results$CV_Error)]]
  bestModel = mda(formula, data, subclasses = bestS)
  bestPrec = ifelse(nclass==2, precisions[bestS],NA_real_)
  
  return(list(model=bestModel, cvPrecision=bestPrec))
  
}

selector <- function(bin_pred, labMap, tieBreak, noneBreak){
  names(bin_pred) <- sub("^bin_","", names(bin_pred))
  
  bin_pred <- bin_pred %>%
    rowwise() %>%
    mutate(
      best0 = {
        values <- as.numeric(as.character(c_across(everything())))
        if (sum(values) == 1) {names(bin_pred)[which(values == 1)]
        } else if (sum(values) == 0) {"none"
        } else {tieBreak
        }
      },
      bestS = if_else(best0 == "none", noneBreak, best0)
    ) %>%
    mutate(
      best0 = factor(labMap[best0],levels = labMap),
      bestS = factor(labMap[bestS],levels = labMap[-length(labMap)])
    ) %>%
    ungroup()
  
  return(bin_pred[,c("best0","bestS")])
}

