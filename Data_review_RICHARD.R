install.packages("BSDA")
library(BSDA)
install.packages("reshape2")
library(reshape2)
install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
install.packages("car")
library(car)
library(MASS)
# Define data

kaggle_training_data <- read.csv("customer_propensity_training_sample.csv")

## EDA
# General data review
dim(kaggle_training_data)
colnames(kaggle_training_data)
summary(kaggle_training_data)   # Ordered mean 0.04193, thus ~4% of users made an order 
                                # User ID should be excluded from the model, unique identifier
str(kaggle_training_data )


# Confirm inputs are binary
binary_columns_check <- c(
  "basket_icon_click", "basket_add_list", "basket_add_detail",
  "sort_by", "image_picker", "account_page_click", "promo_banner_click",
  "detail_wishlist_add", "list_size_dropdown", "closed_minibasket_click", "checked_delivery_detail",
  "checked_returns_detail", "sign_in", "saw_checkout", "saw_sizecharts",
  "saw_delivery", "saw_account_upgrade", "saw_homepage", "device_mobile",
  "device_computer", "device_tablet", "returning_user", "loc_uk",
  "ordered"
)

lapply(kaggle_training_data[columns_to_check], unique)

(unique_values <- unique(sampled_kaggle_training_data_df$UserID))  # Non binary, USERID


# Converting data to relevant categories (all binary to numeric and USERID remains character)
cols_to_numeric <- names(kaggle_training_data)[2:ncol(kaggle_training_data)]
kaggle_training_data[cols_to_numeric] <- lapply(kaggle_training_data[cols_to_numeric], as.numeric)
str(kaggle_training_data )


# Check for Missing Values -> NONE
sum(is.na(kaggle_training_data))  # Data has NO NAs
colSums(is.na(kaggle_training_data)) # Column has NO NAs


# Evaluate DEPENDENT variable
barplot(table(kaggle_training_data$ordered), main = "Distribution of Ordered", names.arg = c("Not Ordered", "Ordered"))


# Evaluate INDEPENDENT variables
for (col in binary_columns_check) {
  barplot(
    table(kaggle_training_data[[col]]),
    main = paste("Distribution of", col), 
    names.arg = c("0", "1")
  )
}

(counts_by_column <- lapply(kaggle_training_data[, binary_columns_check], table))
# Columns which saw 1 distribution > 50k: $basket_add_detail, $list_size_dropdown, $saw_homepage, 
# $device_mobile, $device_computer, $device_tablet, $returning_user, $loc_uk

large_distrubution_columns <- c(
  "basket_add_detail", "list_size_dropdown", 
  "saw_homepage", "device_mobile",
  "device_computer", "device_tablet", 
  "returning_user", "loc_uk"
)

for (col in binary_columns_check) {
  print(paste(col))
  print(aggregate(ordered ~ kaggle_training_data[[col]], data = kaggle_training_data, mean))
}
