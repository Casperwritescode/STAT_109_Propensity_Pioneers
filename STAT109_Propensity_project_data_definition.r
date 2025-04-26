library(SparkR)
sparkR.session()

# Define the file path
training_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_training_sample.csv"
testing_file_path <- "dbfs:/FileStore/tables/ecom_user_data/customer_propensity_testing_sample.csv"

# Read the CSV file into a Spark DataFrame
kaggle_training_data_df <- read.df(training_file_path, source = "csv", header = "true", inferSchema = "true")
kaggle_testing_data_df <- read.df(testing_file_path, source = "csv", header = "true", inferSchema = "true")

# Combine datasets
kaggle_total_data <- rbind(kaggle_training_data_df, kaggle_testing_data_df)

# Bring the data from Spark into R memory - only required for SparkR
kaggle_total_data <- collect(kaggle_total_data)


# Refactor device type variable to be categorical
kaggle_total_data$device_type <- ifelse(
  kaggle_total_data$device_mobile == 1, "mobile",
  ifelse(
    kaggle_total_data$device_computer == 1, "computer",
    "tablet"
  )
)

# Then make sure it's a factor - factor() is similar to as.factor()
kaggle_total_data$device_type <- factor(
  kaggle_total_data$device_type,
  levels = c("mobile", "computer", "tablet")
)

# Check the structure
str(kaggle_total_data)
