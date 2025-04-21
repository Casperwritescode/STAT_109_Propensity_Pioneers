# Databricks notebook source
# MAGIC %md
# MAGIC 50-50 split

# COMMAND ----------

library(SparkR)
sparkR.session()
library(caret)

# Define the file path
training_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_training_sample.csv"
testing_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_testing_sample.csv"

# Read the CSV file into a Spark DataFrame
kaggle_training_data_df <- read.df(training_file_path, source = "csv", header = "true", inferSchema = "true")
kaggle_testing_data_df <- read.df(testing_file_path, source = "csv", header = "true", inferSchema = "true")

# Combine datasets
kaggle_total_data <- rbind(kaggle_training_data_df, kaggle_testing_data_df)

# Register the Spark DataFrame as a temporary table
createOrReplaceTempView(kaggle_total_data, "customer_data")

# Partition the data randomly into train/test
partition_df_5050 <- SparkR::sql("
  SELECT *, 
         CASE WHEN rand(123) < 0.5 THEN 1 ELSE 2 END AS partition 
  FROM customer_data
")

# Split into training and testing sets
train_df_5050 <- SparkR::filter(partition_df_5050, partition_df_5050$partition == 1)
test_df_5050 <- SparkR::filter(partition_df_5050, partition_df_5050$partition == 2)

# Train logistic regression model
final_model <- glm(ordered ~ . -UserID, data = train_df_5050, family = binomial(link = "logit"))

# Predict on test data
test_prediction_5050 <- predict(final_model, test_df_5050)

# Collect predictions and true labels
test_prediction_results_5050 <- SparkR::collect(test_prediction_5050)
test_df_collected_5050 <- SparkR::collect(test_df_5050)

# Step 1: Create binary prediction from logistic probabilities
test_prediction_results_5050$predicted_label <- ifelse(test_prediction_results_5050$prediction > 0.5, 1, 0)

# Step 2: Add the true labels (ordered) to the prediction results
test_prediction_results_5050$ordered <- test_df_collected_5050$ordered

# Step 3: Convert both to factors for caret::confusionMatrix
test_prediction_results_5050$predicted_label <- as.factor(test_prediction_results_5050$predicted_label)
test_prediction_results_5050$ordered <- as.factor(test_prediction_results_5050$ordered)

# Step 4: Evaluate with confusion matrix
cm <- confusionMatrix(data = test_prediction_results_5050$predicted_label,
                      reference = test_prediction_results_5050$ordered,
                      positive = '1')

# Output results
print(cm)

# COMMAND ----------

# MAGIC %md
# MAGIC Define ROC, AUC and optimal threshold

# COMMAND ----------

library(pROC)

# Collect the relevant columns from SparkR dataframe
collected_df_5050 <- collect(select(test_df_5050, "ordered"))
collected_predictions_5050 <- collect(select(test_prediction_5050, "prediction"))

# str(collected_df)
# str(collected_predictions)

# Extract as vectors
response_vector_5050 <- collected_df_5050$ordered
predictor_vector_5050 <- unlist(collected_predictions_5050)

r_train_5050 <- multiclass.roc(response_vector_5050, predictor_vector_5050, percent = TRUE)
roc_train_5050 <- r_train_5050[['rocs']]
r1_train_5050 <- roc_train_5050[[1]]
plot.roc(r1_train_5050, col = 'red', lwd = 5, main = '50-50 Model train data ROC Curve')

plot.roc(r1_train_5050,
         print.auc = T,
         auc.polygon = T,
         max.auc.polygon = T,
         auc.polygon.col = 'lightblue',
         print.thres = T,
         main = '50-50 Model train data ROC Curve + AUC')

# Display AUC value
auc(r1_train_5050)

# Display best threshold
coords(r1_train_5050, "best", ret="threshold", transpose = FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC 60-40 split

# COMMAND ----------

library(SparkR)
sparkR.session()
library(caret)

# Define the file path
training_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_training_sample.csv"
testing_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_testing_sample.csv"

# Read the CSV file into a Spark DataFrame
kaggle_training_data_df <- read.df(training_file_path, source = "csv", header = "true", inferSchema = "true")
kaggle_testing_data_df <- read.df(testing_file_path, source = "csv", header = "true", inferSchema = "true")

# Combine datasets
kaggle_total_data <- rbind(kaggle_training_data_df, kaggle_testing_data_df)

# Register the Spark DataFrame as a temporary table
createOrReplaceTempView(kaggle_total_data, "customer_data")

# Partition the data randomly into train/test
partition_df_6040 <- SparkR::sql("
  SELECT *, 
         CASE WHEN rand(123) < 0.6 THEN 1 ELSE 2 END AS partition 
  FROM customer_data
")

# Split into training and testing sets
train_df_6040 <- SparkR::filter(partition_df_6040, partition_df_6040$partition == 1)
test_df_6040 <- SparkR::filter(partition_df_6040, partition_df_6040$partition == 2)

# Train logistic regression model
final_model <- glm(ordered ~ . -UserID, data = train_df_6040, family = binomial(link = "logit"))

# Predict on test data
test_prediction_6040 <- predict(final_model, test_df_6040)

# Collect predictions and true labels
test_prediction_results_6040 <- SparkR::collect(test_prediction_6040)
test_df_collected_6040 <- SparkR::collect(test_df_6040)

# Step 1: Create binary prediction from logistic probabilities
test_prediction_results_6040$predicted_label <- ifelse(test_prediction_results_6040$prediction > 0.5, 1, 0)

# Step 2: Add the true labels (ordered) to the prediction results
test_prediction_results_6040$ordered <- test_df_collected_6040$ordered

# Step 3: Convert both to factors for caret::confusionMatrix
test_prediction_results_6040$predicted_label <- as.factor(test_prediction_results_6040$predicted_label)
test_prediction_results_6040$ordered <- as.factor(test_prediction_results_6040$ordered)

# Step 4: Evaluate with confusion matrix
cm <- confusionMatrix(data = test_prediction_results_6040$predicted_label,
                      reference = test_prediction_results_6040$ordered,
                      positive = '1')

# Output results
print(cm)

# COMMAND ----------

# MAGIC %md
# MAGIC Define ROC, AUC and optimal threshold

# COMMAND ----------

library(pROC)

# Collect the relevant columns from SparkR dataframe
collected_df_6040 <- collect(select(test_df_6040, "ordered"))
collected_predictions_6040 <- collect(select(test_prediction_6040, "prediction"))

# str(collected_df)
# str(collected_predictions)

# Extract as vectors
response_vector_6040 <- collected_df_6040$ordered
predictor_vector_6040 <- unlist(collected_predictions_6040)

r_train_6040 <- multiclass.roc(response_vector_6040, predictor_vector_6040, percent = TRUE)
roc_train_6040 <- r_train_6040[['rocs']]
r1_train_6040 <- roc_train_6040[[1]]
plot.roc(r1_train_6040, col = 'red', lwd = 5, main = '60-40 Model train data ROC Curve')

plot.roc(r1_train_6040,
         print.auc = T,
         auc.polygon = T,
         max.auc.polygon = T,
         auc.polygon.col = 'lightblue',
         print.thres = T,
         main = '60-40 Model train data ROC Curve + AUC')

# Display AUC value
auc(r1_train_6040)

# Display best threshold
coords(r1_train_6040, "best", ret="threshold", transpose = FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC 70-30 split

# COMMAND ----------

library(SparkR)
sparkR.session()
library(caret)

# Define the file path
training_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_training_sample.csv"
testing_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_testing_sample.csv"

# Read the CSV file into a Spark DataFrame
kaggle_training_data_df <- read.df(training_file_path, source = "csv", header = "true", inferSchema = "true")
kaggle_testing_data_df <- read.df(testing_file_path, source = "csv", header = "true", inferSchema = "true")

# Combine datasets
kaggle_total_data <- rbind(kaggle_training_data_df, kaggle_testing_data_df)

# Register the Spark DataFrame as a temporary table
createOrReplaceTempView(kaggle_total_data, "customer_data")

# Partition the data randomly into train/test
partition_df_7030 <- SparkR::sql("
  SELECT *, 
         CASE WHEN rand(123) < 0.7 THEN 1 ELSE 2 END AS partition 
  FROM customer_data
")

# Split into training and testing sets
train_df_7030 <- SparkR::filter(partition_df_7030, partition_df_7030$partition == 1)
test_df_7030 <- SparkR::filter(partition_df_7030, partition_df_7030$partition == 2)

# Train logistic regression model
final_model <- glm(ordered ~ . -UserID, data = train_df_7030, family = binomial(link = "logit"))

# Predict on test data
test_prediction_7030 <- predict(final_model, test_df_7030)

# Collect predictions and true labels
test_prediction_results_7030 <- SparkR::collect(test_prediction_7030)
test_df_collected_7030 <- SparkR::collect(test_df_7030)

# Step 1: Create binary prediction from logistic probabilities
test_prediction_results_7030$predicted_label <- ifelse(test_prediction_results_7030$prediction > 0.5, 1, 0)

# Step 2: Add the true labels (ordered) to the prediction results
test_prediction_results_7030$ordered <- test_df_collected_7030$ordered

# Step 3: Convert both to factors for caret::confusionMatrix
test_prediction_results_7030$predicted_label <- as.factor(test_prediction_results_7030$predicted_label)
test_prediction_results_7030$ordered <- as.factor(test_prediction_results_7030$ordered)

# Step 4: Evaluate with confusion matrix
cm <- confusionMatrix(data = test_prediction_results_7030$predicted_label,
                      reference = test_prediction_results_7030$ordered,
                      positive = '1')

# Output results
print(cm)

# COMMAND ----------

# MAGIC %md
# MAGIC Define ROC, AUC and optimal threshold

# COMMAND ----------

library(pROC)

# Collect the relevant columns from SparkR dataframe
collected_df_7030 <- collect(select(test_df_7030, "ordered"))
collected_predictions_7030 <- collect(select(test_prediction_7030, "prediction"))

# str(collected_df)
# str(collected_predictions)

# Extract as vectors
response_vector_7030 <- collected_df_7030$ordered
predictor_vector_7030 <- unlist(collected_predictions_7030)

r_train_7030 <- multiclass.roc(response_vector_7030, predictor_vector_7030, percent = TRUE)
roc_train_7030 <- r_train_7030[['rocs']]
r1_train_7030 <- roc_train_7030[[1]]
plot.roc(r1_train_7030, col = 'red', lwd = 5, main = '70-30 Model train data ROC Curve')

plot.roc(r1_train_7030,
         print.auc = T,
         auc.polygon = T,
         max.auc.polygon = T,
         auc.polygon.col = 'lightblue',
         print.thres = T,
         main = '70-30 Model train data ROC Curve + AUC')

# Display AUC value
auc(r1_train_7030)

# Display best threshold
coords(r1_train_7030, "best", ret="threshold", transpose = FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC 80-20 split

# COMMAND ----------

library(SparkR)
sparkR.session()
library(caret)

# Define the file path
training_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_training_sample.csv"
testing_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_testing_sample.csv"

# Read the CSV file into a Spark DataFrame
kaggle_training_data_df <- read.df(training_file_path, source = "csv", header = "true", inferSchema = "true")
kaggle_testing_data_df <- read.df(testing_file_path, source = "csv", header = "true", inferSchema = "true")

# Combine datasets
kaggle_total_data <- rbind(kaggle_training_data_df, kaggle_testing_data_df)

# Register the Spark DataFrame as a temporary table
createOrReplaceTempView(kaggle_total_data, "customer_data")

# Partition the data randomly into train/test
partition_df_8020 <- SparkR::sql("
  SELECT *, 
         CASE WHEN rand(123) < 0.8 THEN 1 ELSE 2 END AS partition 
  FROM customer_data
")

# Split into training and testing sets
train_df_8020 <- SparkR::filter(partition_df_8020, partition_df_8020$partition == 1)
test_df_8020 <- SparkR::filter(partition_df_8020, partition_df_8020$partition == 2)

# Train logistic regression model
final_model <- glm(ordered ~ . -UserID, data = train_df_8020, family = binomial(link = "logit"))

# Predict on test data
test_prediction_8020 <- predict(final_model, test_df_8020)

# Collect predictions and true labels
test_prediction_results_8020 <- SparkR::collect(test_prediction_8020)
test_df_collected_8020 <- SparkR::collect(test_df_8020)

# Step 1: Create binary prediction from logistic probabilities
test_prediction_results_8020$predicted_label <- ifelse(test_prediction_results_8020$prediction > 0.5, 1, 0)

# Step 2: Add the true labels (ordered) to the prediction results
test_prediction_results_8020$ordered <- test_df_collected_8020$ordered

# Step 3: Convert both to factors for caret::confusionMatrix
test_prediction_results_8020$predicted_label <- as.factor(test_prediction_results_8020$predicted_label)
test_prediction_results_8020$ordered <- as.factor(test_prediction_results_8020$ordered)

# Step 4: Evaluate with confusion matrix
cm <- confusionMatrix(data = test_prediction_results_8020$predicted_label,
                      reference = test_prediction_results_8020$ordered,
                      positive = '1')

# Output results
print(cm)

# COMMAND ----------

# MAGIC %md
# MAGIC Define ROC, AUC and optimal threshold

# COMMAND ----------

library(pROC)

# Collect the relevant columns from SparkR dataframe
collected_df_8020 <- collect(select(test_df_8020, "ordered"))
collected_predictions_8020 <- collect(select(test_prediction_8020, "prediction"))

# str(collected_df)
# str(collected_predictions)

# Extract as vectors
response_vector_8020 <- collected_df_8020$ordered
predictor_vector_8020 <- unlist(collected_predictions_8020)

r_train_8020 <- multiclass.roc(response_vector_8020, predictor_vector_8020, percent = TRUE)
roc_train_8020 <- r_train_8020[['rocs']]
r1_train_8020 <- roc_train_8020[[1]]
plot.roc(r1_train_8020, col = 'red', lwd = 5, main = '80-20 Model train data ROC Curve')

plot.roc(r1_train_8020,
         print.auc = T,
         auc.polygon = T,
         max.auc.polygon = T,
         auc.polygon.col = 'lightblue',
         print.thres = T,
         main = '80-20 Model train data ROC Curve + AUC')

# Display AUC value
auc(r1_train_8020)

# Display best threshold
coords(r1_train_8020, "best", ret="threshold", transpose = FALSE)