normalize <- function(x)
{
  return(max(x)- min(x))
} 


#' @title bandwidth selection
#' 
#' @description choosing the best bandwidth through LOOCV 
#' 
#' @param XY 
#' @param method
#' @param kcode
#' @param epsilon
#' @param bw
#' @param N_min
#' 
#' @export bw_select
bw_select <- function(XY, method = 2, kcode = 1,
                      epsilon = 0.05, bw, N_min = 1){
  scale.factor <- apply(as.matrix(XY[,1:ncol(XY)-1]),2, FUN = normalize)
  bw <- bw * scale.factor
  h <- h_select_i(XY, method, kcode, epsilon, bw, N_min)
  h <- h/scale.factor
}

#' @title Local linear regression
#' 
#' @description Multivariate local linear regression 
#' 
#' @param XY 
#' @param method
#' @param kcode
#' @param epsilon
#' @param bw
#' @param N_min
#' 
#' @return h 
#' @export loclin
loclin <- function(XY, method = 2, kcode = 1, 
                   epsilon = 0.05, bw, N_min = 1){
  scale.factor <- apply(as.matrix(XY[,1:ncol(XY)-1]),2, FUN = normalize)
  bw <- bw * scale.factor
  XY.pred <- loclinear_i(XY, method, kcode, epsilon, bw, N_min)
  return (XY.pred)
}

#' @title smoothing of predicting values
#' 
#' @description Predict values through local linear smoothing 
#' 
#' @param XY 
#' @param X_pred
#' @param method
#' @param kcode
#' @param epsilon
#' @param bw
#' @param N_min
#' 
#' @export ll.predict
ll.predict <- function(XY, X_pred, method = 2, kcode = 1, epsilon = 0.05,
                       bw, N_min = 1 ){
  scale.factor <- apply(as.matrix(XY[,1:ncol(XY)-1]),2, FUN = normalize)
  bw <- bw * scale.factor
  predict.values <- return(predict_i(XY, X_pred, method, kcode, epsilon, bw, N_min))
  bw <- bw / scale.factor
  predict.values
}

partition.data<- function(XY,k){
  set.seed(123)
  XY <- data.frame(XY)
  if (k != nrow(XY)){
    XY <- dplyr::mutate(XY, my.folds = sample(1:k,
                   size = nrow(XY),
                   replace = TRUE))
  }
  else {
    XY <- dplyr::mutate(XY, my.folds = sample(1:k,
                                              size = nrow(XY),
                                              replace = FALSE))  
  }
  XY
}

#' @title kfold cross valdation
#' 
#' @description choosing the best bandwidth through kfold cross validation
#' 
#' @param XY, 
#' @param method
#' @param kcode
#' @param epsilon
#' @param bw
#' @param N_min
#' @param k 
#' 
#' @importFrom Tidyverse dplyr
#' 
#' @export cv.kfold
cv.kfold <- function(XY, method = 2, kcode = 1, epsilon =0.05, bw, N_min = 1, k = 5 ){
    XY = partition.data(XY,k)
    SSE = 0
    best_CV = -1
    for (j in 1:nrow(bw)){
      h = bw[j, ]
      tot_CV = 0 
      for (i in 1:k){
        train <- subset(XY, my.folds != i)
        train = train[,1:(ncol(train)-1)]
        train = as.matrix(train)
        test <- subset(XY, my.folds == i)
        Y_val = test[,ncol(test)-1]
        test = test[,1:(ncol(test)-2)] # 2 (1 for my.folds, 1 for Y)
        test = as.matrix(test) 
        Y_val_predict = ll.predict(train, test, method, kcode, epsilon, h, N_min)
        CV <- (Y_val - Y_val_predict)^2
        tot_CV <- tot_CV + mean(CV)
      }
      SSE = cbind(SSE,tot_CV)
      if(best_CV == -1){ 
        best_CV = tot_CV
      }
      if (tot_CV < best_CV) {
        best_CV = tot_CV 
        best_h = h
      }
    }
  best_h
}

