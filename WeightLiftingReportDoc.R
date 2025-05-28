
# 2. Data Preprocessing
# 2.1 Loading and Initial Exploration
library(caret)
library(randomForest)
library(dplyr)
# Download training and test data
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(train_url, "pml-training.csv", method = "curl")
download.file(test_url, "pml-testing.csv", method = "curl")


# Load data
train_data <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test_data <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))

# Check dimensions
dim(train_data)  # 19622 x 160
dim(test_data)   # 20 x 160
# Data Cleaning 
# Remove columns with >95% NA values
train_data <- train_data[, colMeans(is.na(train_data)) < 0.95]
test_data <- test_data[, colMeans(is.na(test_data)) < 0.95]

# Remove metadata columns (non-predictive)
cols_to_remove <- c("X", "user_name", "raw_timestamp_part_1", 
                    "raw_timestamp_part_2", "cvtd_timestamp",
                    "new_window", "num_window")
train_data <- train_data %>% select(-all_of(cols_to_remove))
test_data <- test_data %>% select(-all_of(cols_to_remove))

# Convert classe to factor
train_data$classe <- factor(train_data$classe, levels = c("A", "B", "C", "D", "E"))

# Ensure all remaining columns are numeric
numeric_cols <- sapply(train_data, is.numeric)
train_data <- train_data[, numeric_cols | names(train_data) == "classe"]
test_data <- test_data[, numeric_cols[names(numeric_cols) %in% names(test_data)]]
# Data Partitioning
set.seed(123)
train_idx <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training <- train_data[train_idx, ]
validation <- train_data[-train_idx, ]
# Model Training (Random Forest)
# Train model with cross-validation
ctrl <- trainControl(method = "cv", number = 5)
model_rf <- train(classe ~ ., 
                  data = training, 
                  method = "rf",
                  trControl = ctrl,
                  ntree = 100,
                  importance = TRUE)

# View model details
print(model_rf)
# Model Evaluation
pred_val <- predict(model_rf, validation)
confusionMatrix(pred_val, validation$classe)
# Out-of-Sample Error Estimate
Estimated out-of-sample error: 0.7% (1 - accuracy)
# Final Predictions on Test Set
test_pred <- predict(model_rf, test_data)
test_pred