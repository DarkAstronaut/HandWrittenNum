# Read train and test data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Check if the data is uniformly distributed over all labels
label.freq = table(train$label)
#barplot(label.freq)

# Check if NAs exist:
which(is.na(train))
# No NAs present

# convert lable to a categorical variable
train$label <- as.factor(train$label)

# separate pixels from the training and testing sets
train_pixel <- train[,2:ncol(train)]
test_pixel <- test[,2:ncol(test)]


## Excluding highly correlated variables
# Obtaining Correlation matrix
MatCor <- cor(train[sapply(train,is.numeric)])

# The standard deviation is zero, hence correlation matrix cannot be used

# Use nearZeroVar to preprocess the pixels
library(caret)

#zeroVar <- as.data.frame(colnames(train_pixel[nearZeroVar(train_pixel)]))

train_nzv <- nearZeroVar(train_pixel)
test_nzv <-  nearZeroVar(test_pixel)
# nearZeroVar selects 532 out of 784 factors/pixels

# create a new dataframe with these 532 factors. 
# This dataframe will be used for further preprocessing

train_postnzv <- train_pixel[,-train_nzv]
test_postnzv <- train_pixel[,-train_nzv]

# Preprocess using PCA
train_preProcValues <- preProcess(train_postnzv, method=c("pca"))
test_preProcValues <- preProcess(test_postnzv, method = c("pca"))

x_trainTransformed <- predict(train_preProcValues, train_postnzv)
x_testTransformed <- predict(test_preProcValues, test_postnzv)


dim(x_trainTransformed)
dim(x_testTransformed)



### Building Models 

# 1. Multiclass logistic regression
train_final <- x_trainTransformed
train_final$label <- train$label

dim(train_final)

LR_Model <- glm(label~., 
                family=binomial(link='logit'), data=train_final)



View(LR_Model$y)
summary(LR_Model$y)
LR_Fitted <- predict(LR_Model, newdata= x_testTransformed, type = "response")
View(LR_Fitted)

table <- table(train_final$label, LR_Fitted >0.5)



LR_confusionMatrix <- confusionMatrix(LR_Fitted, train_final$label)
LR_confusionMatrix
str(train_final$label)
str(LR_Fitted)
View(LR_Fitted)
LR_Fitted <- as.data.frame(LR_Fitted)

#Error in confusionMatrix.default(LR_Fitted, train_final$label) : 
#the data cannot have more levels than the reference

table(factor(LR_Fitted, levels=min(x_testTransformed):max(x_testTransformed)),
      factor(x_testTransformed, levels=min(x_testTransformed):max(x_testTransformed)))


LR_accuracy <- as.numeric(LR_confusionMatrix$overall["Accuracy"])



# Random forests:
library(randomForest)

samplerows <- sample(1:nrow(train_pixel), nrow(train)*0.6, replace=FALSE)
train_rf <- x_trainTransformed[samplerows,]
test_rf <- x_trainTransformed[-samplerows,]

train_labels <- as.factor(train[samplerows,]$label)
test_labels<- as.factor(train[-samplerows]$labels)


RF_Model <- randomForest(train_rf, train_labels, ntree = 100)
predict_labels <- predict(RF_Model, test_rf)

accuracy <- sum(diag(RF_Model$confusion))*100/sum(RF_Model$confusion)
accuracy
## confusion matrix 
RF_Model$confusion

