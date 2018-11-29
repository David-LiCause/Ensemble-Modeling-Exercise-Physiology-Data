Ensemble Predictive Modeling with Exercise Physiology Data
================

Analysis Summary
----------------

The following analysis examines exercise physiology data from accelerometers from 6 study participants. The data was pulled from <http://groupware.les.inf.puc-rio.br/har> with permission. The following predictive modeling comprises of an ensemble approach to distinguish between 5 different barbell exercises. Models were genereated using random forest, gradient boosting and support vector machines. These 3 models were ensembled to generate a &gt;99% classification accuracy on a out of sample test set.

Data Pre-Processing
-------------------

The data includes 19,622 observations across 160 variables. Missing values were converted to NA and relevant predictor variables were extracted for the purpose of modeling. Additionally, columns with &gt;95% missing data were removed prior to modeling.

``` r
# Import data
train_master <- read.csv("pml-training.csv", header=T)
test_master <- read.csv("pml-testing.csv", header=T)

# Recode missing values as NA
train_master[train_master=="#DIV/0!"] <- NA
train_master[train_master==""] <- NA

# Remove columns that have >95% NA values
missing_cols <- which(apply(train_master, 2, function(x) sum(is.na(x)) > .95*nrow(train_master)))
train_master <- train_master %>%
  select(-missing_cols)
```

Partition data into train, test and validation sets
---------------------------------------------------

An out of sample test data set (n = 3925) was partitioned from the training set through random sampling. The remaining 15,697 observations in the training set were randomly partitioned into 10 folds to prep for 10-fold cross validation.

``` r
# Create vector of predictor colnames
pred_vars <- colnames(train_master)[8:59]

set.seed(123)
# Create train, validation and test sets from training data
train_id <- sample(1:nrow(train_master), .8*nrow(train_master), replace=F)
train <- train_master[train_id,]
test <- train_master[-train_id,]
k <- sample(rep(c(1:10), ((nrow(train))/10)+10))[1:nrow(train)]
train <- cbind(k, train) %>%
  as.data.frame()
```

Tune model hyperparameters using 10-fold cross validation
---------------------------------------------------------

10-fold cross validation was performed on the training data set and error was measured for each type of model (random forest, gradient boosting, support vector machine and ensemble model). Since the outcome contained well balanced classes the overall classification accuracy was used to compare model performance.

``` r
cv_error <- matrix(NA, max(k), 5)
colnames(cv_error) <- c("k", "random_forest_accuracy", "gradient_boosting_accuracy", 
                        "support_vector_machine_accuracy", "ensemble_accuracy")

for (i in 1:max(k)) {
  train_x <- train[train$k!=i, pred_vars]
  train_y <- train[train$k!=i, "classe"]
  val_x <- train[train$k==i, pred_vars]
  val_y <- train[train$k==i, "classe"]
  # Build models
  rf_model <- randomForest(train_x, train_y)
  gbm_model <- gbm(train_y ~ ., data=as.data.frame(cbind(train_y, train_x)), distribution = "multinomial")
  svm_model <- svm(train_x, train_y, kernel='linear')
  # Join validation set predictions for ensembling
  rf_val_pred <- predict(rf_model, newdata=val_x)
  gbm_val_pred_mat <- predict(gbm_model, newdata=val_x, n.trees = 100)
  gbm_val_pred <- apply(gbm_val_pred_mat, 1, which.max)
  gbm_val_pred <- factor(gbm_val_pred, levels = c(1,2,3,4,5), labels = c("A", "B", "C", "D", "E"))
  svm_val_pred <- predict(svm_model, newdata=val_x)
  pred_matrix <- data.frame(y=val_y, rf=rf_val_pred, gbm=gbm_val_pred, svm=svm_val_pred)
  # Ensemble models 
  ens_model <- multinom(y~., data=pred_matrix)
  ens_val_pred <- predict(ens_model, newdata=pred_matrix)
  # Measure error
  cv_error[i,1] <- i
  cv_error[i,2] <- sum(as.numeric(rf_val_pred==val_y))/length(val_y)
  cv_error[i,3] <- sum(as.numeric(gbm_val_pred==val_y))/length(val_y)
  cv_error[i,4] <- sum(as.numeric(svm_val_pred==val_y))/length(val_y)
  cv_error[i,5] <- sum(as.numeric(ens_val_pred==val_y))/length(val_y)
}

# Save models to local machine
save(rf_model, gbm_model, svm_model, ens_model, file="models.RData")
```

``` r
print(cv_error)
```

    ##        k random_forest_accuracy gradient_boosting_accuracy
    ##  [1,]  1              0.9949012                  0.8260038
    ##  [2,]  2              0.9955414                  0.8146497
    ##  [3,]  3              0.9955442                  0.8376830
    ##  [4,]  4              0.9968214                  0.8232676
    ##  [5,]  5              0.9961710                  0.8359923
    ##  [6,]  6              0.9968153                  0.8108280
    ##  [7,]  7              0.9949109                  0.8155216
    ##  [8,]  8              0.9955386                  0.8221797
    ##  [9,]  9              0.9955329                  0.8021698
    ## [10,] 10              0.9923518                  0.8228171
    ##       support_vector_machine_accuracy ensemble_accuracy
    ##  [1,]                       0.7979605         0.9949012
    ##  [2,]                       0.7955414         0.9955414
    ##  [3,]                       0.7905792         0.9968173
    ##  [4,]                       0.7800381         0.9968214
    ##  [5,]                       0.8130185         0.9961710
    ##  [6,]                       0.7789809         0.9968153
    ##  [7,]                       0.7805344         0.9949109
    ##  [8,]                       0.7858509         0.9955386
    ##  [9,]                       0.7823867         0.9955329
    ## [10,]                       0.7756533         0.9923518

From the results of cross validation it's clear that the random forest model architecture outperforms both the gradient boosting and support vector machine models. The ensemble model performance is identical to the random forest classification accuracy due to the ensemble model being trained on the class labels predicted by each model. A possible improvement to the ensemble model could be achieved by training the model using the softmax probabilities from each of the other models.

Get estimate for generalizability of model
------------------------------------------

The out of sample test set was not used in the hyperparameter tuning process in the modeling and is therefore an unbiased measure of generalizability.

``` r
test_x <- test[, pred_vars]
test_y <- test[, "classe"]

# Use models generated in the last iteration of k-fold cross validation to produce predictions for test set
rf_test_pred <- predict(rf_model, newdata=test_x)
gbm_test_pred_mat <- predict(gbm_model, newdata=test_x, n.trees = 100)
gbm_test_pred <- apply(gbm_test_pred_mat, 1, which.max)
gbm_test_pred <- factor(gbm_test_pred, levels = c(1,2,3,4,5), labels = c("A", "B", "C", "D", "E"))
svm_test_pred <- predict(svm_model, newdata=test_x)
pred_test_matrix <- data.frame(y=test_y, rf=rf_test_pred, gbm=gbm_test_pred, svm=svm_test_pred)
ens_test_pred <- predict(ens_model, newdata=pred_test_matrix)

# Get generalizability estimate by examining the accuracy of the models on the test data set
test_accuracy <- sum(as.numeric(ens_test_pred==test_y))/length(test_y)
test_accuracy
```

    ## [1] 0.9941401

The test set accuracy is very similar to the validation set accuracy measured in cross validation. This suggests that the ensemble model is generalizable and is not overfitting the training data.

``` r
# Examine test set error across classes
table(Predicted = ens_test_pred, Actual = test_y)
```

    ##          Actual
    ## Predicted    A    B    C    D    E
    ##         A 1099    5    0    0    0
    ##         B    0  735    9    0    0
    ##         C    0    0  708    8    0
    ##         D    0    0    1  641    0
    ##         E    0    0    0    0  719

An examination of the error rate across classes shows few mislabeled predictions. There model performance does not vary significantly across different classes of the outcome variable.
