
library(dplyr)
library(randomForest)
library(gbm)
library(e1071)
library(caret)
library(nnet)

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

print(cv_error)

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

# Examine test set error across classes
table(Predicted = ens_test_pred, Actual = test_y)

