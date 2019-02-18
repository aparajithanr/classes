### This file is part of the Own Project for the last course
### HarvardX: PH125.9x Data Science: Capstone
### 
### https://github.com/aparajithanr/classes
###
### The execution of the code takes time and could be
### intensive in the use of computing resources
###
### If you already have the "edx" dataframes cleaned,
### save them in an object "edx_cleaned.Rda", or 
### the code will try to download the source file(s) again.

### Construction of the datasets

if (file.exists("edx_cleaned.Rda")) {
    load("edx_cleaned.Rda")
} else {
    # The following code was provided by the instructors:
    
    #############################################################
    # Create edx set
    #############################################################
    
    # Note: this process could take a couple of minutes
    
    if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
    if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
    if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
    if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")

    # Mice Cortex Nuclear 1080 dataset:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls

    # Task #1: Downloading if the cleaned data set is not found in the working directory
    URL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
    fileName <- "/Data_Cortex_Nuclear.xls"
    download.file(URL, paste(getwd(),fileName, sep = ""), mode="wb")

    # Please place the file directly under your documents directory or wherever the getwd() is showing below
    print("The XLS file is placed under directory:")
    print(paste(getwd(),fileName, sep = ""))
    
    library(readxl)
    edx <- read_excel(paste(getwd(),fileName, sep = ""))

    # Task #2: Data Exploration
    print(dim(edx))
    
    print(names(edx))
    
    # Task #3: Data Cleaning
    edx_cleaned <- edx %>% mutate(MouseID = as.numeric(as.factor(MouseID)), Genotype = as.numeric(as.factor(Genotype)), Treatment = as.numeric(as.factor(Treatment)), Behavior = as.numeric(as.factor(Behavior)), class = as.factor(class))

    for(i in 2:ncol(edx_cleaned)-1){
        edx_cleaned[is.na(edx_cleaned[,i]), i] <- 0
    }

    # To make sure that there is zero NA values
    sum(is.na(edx_cleaned))

    # Task #4: Restoring Cleaned Data for future reuse
    ## The next line was introduced by me to avoid creating the datasets every time
    save(edx_cleaned, file = "edx_cleaned.Rda")
}

### Loading packages
library(randomForest)
library(tidyverse)
library(caret)

# Task #5: Data Partioning into Training (70%) & Test (30%) datasets
ind <- sample(2, nrow(edx_cleaned), replace=TRUE, prob=c(0.7,0.3))
trainData_x <- edx_cleaned[ind == 1,] %>% dplyr::select(-MouseID, -Genotype, -Treatment, -Behavior)
trainData_y <- as.factor(trainData_x$class)
trainData_x <- trainData_x %>% dplyr::select(-class)
testData_x <- edx_cleaned[ind == 2,] %>% dplyr::select(-MouseID, -Genotype, -Treatment, -Behavior)
testData_y <- as.factor(testData_x$class)
testData_x <- testData_x %>% dplyr::select(-class)

# Task #6: Train ML model for the entire set of features using RandomForest algorithm
model_rf <- randomForest(trainData_y ~. , data = trainData_x, proximity=TRUE)
confusionMatrix(predict(model_rf), trainData_y)

# Task #7: Identify the subset of proteins which are discriminant by variable importance > 10
varImpPlot(model_rf, main='Variable Importance Plot: Base Model')
imp <- importance(model_rf)
protein_imp <- data.frame(protein = rownames(imp), meanVal = imp)
protein_iv_10 <- protein_imp[which(protein_imp[order(-protein_imp$MeanDecreaseGini),]$MeanDecreaseGini > 10),]$protein
iv.used_10 <- length(protein_iv_10)

# Task #8: Train ML model only for the subset of proteins which are discriminant
trainData_x_iv_10 <- trainData_x %>% dplyr::select(protein_iv_10)
model_rf_iv_10 <- randomForest(trainData_y ~., data=trainData_x_iv_10, proximity=TRUE)
confusionMatrix(predict(model_rf_iv_10), trainData_y)

# Task #9: Run predictions on the test data using the trained model
testData_x_iv_10 <- testData_x %>% dplyr::select(protein_iv_10)
predictions_test_y_iv_10 <- predict(model_rf_iv_10, testData_x_iv_10)
confusionMatrix(predictions_test_y_iv_10, testData_y)
ggplot(testData_x_iv_10)+
    geom_point(aes(y=testData_y, x=predictions_test_y_iv_10, color=as.factor(testData_y)))+
    ylab('Actual')+
    xlab('Predicted')+
    ggtitle('Actual vs Predicted - Feature Selection by Variable Importance')+
    geom_abline(colour="grey")

# Task #10: Applying PCA to cross verify the selection of the subset of proteins which are discriminant
trainData_x_pca <- prcomp(trainData_x, center = TRUE, scale. = TRUE)

# Task #11: Exploring PCA and Variance Explained
summary(trainData_x_pca)

str(trainData_x_pca)

var_explained <- cumsum(trainData_x_pca$sdev^2/sum(trainData_x_pca$sdev^2))

# Task #12: Identify the subset of proteins which are discriminant by PCA with variance explained > 90%
pc.used_90 <- which(var_explained>=0.90)[1]
trainData_x_pc_90 <- trainData_x[,1:pc.used_90]
protein_pc_90 <- colnames(trainData_x_pc_90)

# Task #13: Train ML model for the subset of proteins (identified by PCA) using RandomForest algorithm
model_rf_pc_90 <- randomForest(trainData_y ~., data=trainData_x_pc_90, proximity=TRUE)
confusionMatrix(predict(model_rf_pc_90), trainData_y)

# Task #14: Run predictions on the test data using the trained model
testData_x_pc_90 <- testData_x[,1:pc.used_90]
predictions_test_y_pc_90 <- predict(model_rf_pc_90, testData_x_pc_90)
confusionMatrix(predictions_test_y_pc_90, testData_y)
ggplot(testData_x_pc_90)+
    geom_point(aes(y=testData_y, x=predictions_test_y_pc_90, color=as.factor(testData_y)))+
    ylab('Actual')+
    xlab('Predicted')+
    ggtitle('Actual vs Predicted - Feature Selection by PCA')+
    geom_abline(colour="grey")

# Task #15: Obtain results by comparing the top 20 discriminant proteins identified by both processes
sum(head(protein_iv_10, 20) == head(protein_pc_90, 20))

head(protein_pc_90, 20)
