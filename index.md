---
title: "Practical Machine Learning (PML) Course Project"
author: "Bo Suzow"
date: "December 18, 2017"
output:
   html_document:
      keep_md: TRUE
   
---
*** 



## Introduction

The purpose of the data analysis discussed in this report (as the PML course project) is to build a model to accurately predict the weight lifting exercise classification (A through E).  

The test dataset is from [the Weight Lifting Exercises research](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).  The research team (Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.) collected data from 6 male paritipants (each wearing 4 sensors) who were asked to perform the unilateral dumbbell biceps curl in 5 different ways classified in A through E where class A is the correct way. The research paper is available [here](http://web.archive.org/web/20170519033209/http://groupware.les.inf.puc-rio.br:80/public/papers/2013.Velloso.QAR-WLE.pdf). 

***

## Data Load

The training and test sets are downloaded from the specific locations provided by the instructor. The test set hosts 20 observations for which class predictions are to be made with the model built in this report.


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","HARtrain.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","HARtest.csv")

train = read.csv("HARtrain.csv")
test  = read.csv("HARtest.csv")

dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```


A large number of predictors (160) are in the dataset. Next, we will find out if there are missing values. 



```r
sum(is.na(train)) / (nrow(train)*ncol(train))
```

```
## [1] 0.4100856
```

```r
sum(is.na(test))  / (nrow(test)*ncol(test))
```

```
## [1] 0.625
```


A substantial portion of each set consists of missing values.  The missing values in the test dataset are problematic as they hinder the predictive model constructed based on the training set from being  operated on the test set. Hence, the predictors with NAs in the training and test sets are removed. 

###### (Note: This type of missing value treatment is not a best practice all. However, it is employed as the test set was given by the instructor with missing values. Normally, the "cleaned-up" test set gets created from the sample dataset via a data slicing scheme.)   

***

## Data Cleaning


```r
tr=train[,!sapply(train,function(x) any(is.na(x)))]
te=test[,!sapply(test,function(x) any(is.na(x)))]
```


The numbers of predictors between two sets are not in agreement after the NA removals. 



```r
sum(sapply(tr,function(x) x=="")) / (nrow(tr) * ncol(tr))
```

```
## [1] 0.3474967
```

```r
sum(sapply(te,function(x) x=="")) / (nrow(te) * ncol(te))
```

```
## [1] 0
```
Null values account for 35% of the training dataset. They are observed in continuous predictors. The test set shows no null values. The null values are removed from the training set. 


```r
tr=tr[,!sapply(tr,function(x) any(x==""))]
dim(tr)
```

```
## [1] 19622    60
```

One last step of the data preparation is to remove the first 7 predictors from each set as they play identification or timestamp roles. The classification prediction should not depend on partipant names nor activity times. A predictive model is to be built only on   measurements.


```r
tr = tr[,c(8:ncol(tr))]
te = te[,c(8:ncol(te))]
```

***

## Exploratory Data Analysis

50+ predictors that survived the data cleaning are still large for sensible plots.  We pick and choose a few predictors to explore data by class (variable name: classe)  



```r
# Plot the 'total' measurement predictors 

tr.total = tr %>% select(grep("total",names(tr)),classe)

tr.total.melt = melt(tr.total)
g.totals.melt = ggplot(tr.total.melt, aes(x=classe,y=value, color=classe)) + 
                geom_boxplot() + 
                facet_grid(.~variable) +
                ggtitle("Boxplot of 'total' type measurements")

g.totals.melt
```

<img src="index_files/figure-html/EDA-1.png" style="display: block; margin: auto;" />

```r
# Plot some measurements from the belt sensor.

tr.belt = tr %>% select(grep("^accel_belt",names(tr)),classe)

tr.belt.melt = melt(tr.belt)
g.belts.melt = ggplot(tr.belt.melt, aes(x=classe,y=value, color=classe)) + 
                geom_boxplot() + 
                facet_grid(.~variable) +
                ggtitle("Boxplot of accelerometer readings from the belt sensor")
g.belts.melt
```

<img src="index_files/figure-html/EDA-2.png" style="display: block; margin: auto;" />

```r
# Plot some measurements from the arm sensor.

tr.arm = tr %>% select(grep("^gyros_arm",names(tr)),classe)

tr.arm.melt = melt(tr.arm)
g.arm.melt = ggplot(tr.arm.melt, aes(x=classe,y=value, color=classe)) + 
                geom_boxplot() + 
                facet_grid(.~variable) +
                ggtitle("Boxplot of gyroscope readings from the arm sensor")

g.arm.melt
```

<img src="index_files/figure-html/EDA-3.png" style="display: block; margin: auto;" />

```r
# Plot the measurements from the dumbbell sensor.

tr.dumbbell = tr %>% select(grep("^magnet_dumbbell",names(tr)),classe)

tr.dumbbell.melt = melt(tr.dumbbell)
g.dumbbell.melt = ggplot(tr.dumbbell.melt, aes(x=classe,y=value, color=classe)) + 
                geom_boxplot() + 
                facet_grid(.~variable) +
                ggtitle("Boxplot of magnetometer readings from the dumbbell sensor")

g.dumbbell.melt
```

<img src="index_files/figure-html/EDA-4.png" style="display: block; margin: auto;" />

***

## Data Modeling

In absence of a good strategy of predictor selection, all measurement predictors are used in model building. 
The random forest (rf) algorithm is selected for its ability to handle a large number of predictors. With the caret package's trainControl(), cross validation folds can be created with a parameter setting. The 5-fold cross validation is utilized. In order to achieve an efficient execution, [the parallel processing (courtesy of Len Greski)](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md) is incorporated.



```r
library(parallel)    # for parallel processing configuration
library(doParallel)  # for parallel processing configuration

cluster = makeCluster(detectCores() -1 )  # reserve 1 core for OS
registerDoParallel(cluster)

set.seed(100)
model.rf = train(classe~.,data=tr,
               method="rf",
               trControl = trainControl(
                  method="cv",
                  number=5,  # 5-fold CV's
                  allowParallel = TRUE,  # for parallel processing
                  verboseIter = FALSE    # change it to TRUE for verbose mode
               ))

# de-register parallel processing cluster

stopCluster(cluster)
registerDoSEQ()
```

***

## Out of Sample Error Calculation

The accuracy of each fold is at a little over 99%.
The expected out of sample error is less than **1%**. 


```r
# display accuracy information

model.rf$resample[1]
```

```
##    Accuracy
## 1 0.9938854
## 2 0.9931210
## 3 0.9954117
## 4 0.9949058
## 5 0.9949019
```

```r
# calcuate the expected out of sample error

mean(1-model.rf$resample[,1])
```

```
## [1] 0.005554868
```

***

## Prediction

Using the model, the 20 cases in the test set (object name: te) are predicted as follows:


```r
predict(model.rf,te)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

