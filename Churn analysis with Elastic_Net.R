
install.packages("glmnet")
install.packages("tidyverse")

## Import necessary packages in building the model
library(glmnet) 
library(tidyverse)
library(mlr)

rm(list = ls())
# Set working directory
setwd("E://Telecom//Project//Project Data")

## load the csv dataset 
inputData = read.csv("Data.csv",header = TRUE)

## Stratified sampling
stratTrain <- data.frame()
stratTest <- data.frame()
strData <- inputData

for(i in 0:1){
  set <- strData[strData$target==i,]
  rows <- seq(1,nrow(set),1)
  set.seed(1234)
  trainRows <- sample(rows,nrow(set)*0.7)
  trainData <- set[trainRows,]
  testData <- set[-trainRows,]
  stratTrain <- rbind(stratTrain,trainData)
  stratTest <- rbind(stratTest,testData)
}

churnTrainData <- stratTrain[,!names(stratTrain) %in% c("target")]
churnTestData <- stratTest[,!names(stratTest) %in% c("target")]
churnTrainData <- data.frame(target = stratTrain$target, churnTrainData)
churnTestData <- data.frame(target = stratTest$target, churnTestData)

## create a task
trainTask <- makeClassifTask(data = churnTrainData, target = "target", positive = 1)
testTask <- makeClassifTask(data = churnTestData, target = "target" , positive = 1)

##normalize the variables
trainTask <- normalizeFeatures(trainTask,method = "standardize")
testTask <- normalizeFeatures(testTask,method = "standardize")

##glmnet for elasticnet regression
getParamSet("classif.glmnet")
glmLearner <- makeLearner("classif.glmnet",predict.type = "prob")
glmLearner$par.set

glmPS = makeParamSet(
  makeDiscreteParam("alpha", values = c(0.1,0.3,0.5,0.7,0.9)),
  makeNumericParam("lambda", lower =0, upper =1)
)

## random serach for 200 iterations
ranControl <- makeTuneControlRandom(maxit = 200L)

## set 5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 5L)

## hypertuning
set.seed(5555)
glm_tune <- tuneParams(learner = glmLearner,
                       resampling = set_cv,
                       task = trainTask,
                       par.set = glmPS,
                       control = ranControl,
                       measures = list(mmce),
                       show.info = TRUE)

glm_tune$x

glmTunedLearner=setHyperPars(glmLearner,par.vals = glm_tune$x)

glmnmlr=mlr::train(glmTunedLearner,trainTask)
predmlr=predict(glmnmlr,testTask)
predmlr

##mets=list(auc,bac,tpr,tnr,mmce,ber,fpr,fnr)
mets=list(bac,tpr,tnr,mmce,ber,fpr,fnr)
performance(predict(glmnmlr,testTask), measures =mets)

## building confusion matrix for test data
table(churnTestData$target)
ConfusionTable <- table(churnTestData[,1], predmlr$data$response)
ConfusionTable

ConfusionMatrix <- matrix(c(ConfusionTable[4],
                            ConfusionTable[2],
                            ConfusionTable[3],
                            ConfusionTable[1]),
                          nrow = 2,
                          ncol = 2,
                          byrow = TRUE
)

dimnames(ConfusionMatrix) = list(c("Churn", "Non Curn"),         
                                 c("Churn", "Non Curn"))

ConfusionMatrix
table(churnTestData$target)

churn_Accuracy = sum(diag(ConfusionMatrix))/nrow(churnTestData)
churn_Recall = ConfusionMatrix[1,1]/sum(ConfusionMatrix[1,])
churn_Precision = ConfusionMatrix[1,1]/sum(ConfusionMatrix[,1])
churn_Specificity = ConfusionMatrix[2,2]/sum(ConfusionMatrix[2,])

print(c("accuracy"=churnAccuracy,
        "precision"=churn_Precision,
        "recall"=churn_Recall,
        "specificity"=churn_Specificity))
