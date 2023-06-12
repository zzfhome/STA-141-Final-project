# Required packages
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(e1071)

# Read the data
session=list()
for(i in 1:18){
  session[[i]]=readRDS(paste('./Data/session/session',i,'.rds',sep=''))
  print(session[[i]]$mouse_name)
  print(session[[i]]$date_exp)
}

# Part 1: Exploratory data analysis
print("==== Exploratory Data Analysis ====")

# Iterating over sessions
for (i in 1:18) {
  cat("\nSession", i, "\n")
  
  # Get the current session
  current_session = session[[i]]
  
  # Number of neurons
  cat("Number of Neurons: ", length(current_session$spks), "\n")
  
  # Number of trials
  cat("Number of Trials: ", length(current_session$feedback_type), "\n")
  
  # Stimuli conditions (left and right contrasts)
  cat("Stimuli Conditions: Left Contrast Mean: ", mean(current_session$contrast_left), 
      ", Right Contrast Mean: ", mean(current_session$contrast_right), "\n")
  
  # Feedback types
  cat("Feedback Types: Success: ", sum(current_session$feedback_type == 1), 
      ", Failure: ", sum(current_session$feedback_type == -1), "\n")
  
  # Spike counts per trial
  spike_counts = sapply(current_session$spks, function(x) sum(x))
  cat("Spike Counts per Trial: Min: ", min(spike_counts), ", Max: ", max(spike_counts), 
      ", Mean: ", mean(spike_counts), "\n")
}

# Part 2: Data Integration
print("\n==== Data Integration ====")

# Create a new list to hold the processed sessions
processed_sessions <- list()

# Iterate over each session and process the data
for (i in 1:length(session)) {
  
  # Extract each session's data
  current_session <- session[[i]]
  
  # For each trial, compute the mean spike count and add it to the current_session data
  mean_spike_counts <- sapply(current_session$spks, function(x) mean(unlist(x)))
  current_session$mean_spike_count <- mean_spike_counts
  
  # Compute the adjusted number of neurons and add it to the current_session data
  mean_num_neurons = mean(sapply(current_session$spks, function(x) length(unlist(x))))
  adjusted_neurons = sapply(current_session$spks, function(x) length(unlist(x))/mean_num_neurons)
  current_session$adjusted_neurons <- adjusted_neurons
  
  # Add the processed session data to the processed_sessions list
  processed_sessions[[i]] <- current_session
}

# Combine the processed session data
# all_sessions <- bind_rows(processed_sessions)

# Part 3: Model Training and Prediction
print("\n==== Model Training and Prediction ====")

# Preparing the data for modelling
model_data = all_sessions %>% 
  select(feedback_type, contrast_left, contrast_right, mean_spike_count, adjusted_neurons)

# Splitting the data into training and testing sets
set.seed(123)
training_rows = createDataPartition(model_data$feedback_type, p = 0.8, list = FALSE)
training_data = model_data[training_rows, ]
testing_data = model_data[-training_rows, ]

# Train a SVM model on the training data
model = svm(feedback_type ~ ., data = training_data)

# Make predictions on the test data
predictions = predict(model, newdata = testing_data[, -1])

# Section 5: Prediction performance on the test sets

print("\n==== Prediction performance on the test sets ====")

# Evaluate the performance of the model
# Libraries
library(tidyverse)
library(caret)
library(pROC)

# Load the trained model
model <- readRDS("./model.rds")

# Load the test data
test_data1 <- readRDS("./test/test1.rds")
test_data2 <- readRDS("./test/test2.rds")

# Add the combined contrast to the test data
test_data1$contrast_combined <- (test_data1$contrast_left + test_data1$contrast_right) / 2
test_data2$contrast_combined <- (test_data2$contrast_left + test_data2$contrast_right) / 2

# Predict on the test set
predictions1 <- predict(model, newdata = test_data1, type = "response")
predictions2 <- predict(model, newdata = test_data2, type = "response")

# Convert predictions to class labels based on threshold 0.5
predicted_class1 <- ifelse(predictions1 > 0.5, 1, 0)
predicted_class2 <- ifelse(predictions2 > 0.5, 1, 0)

# Convert to factors and ensure levels match
actual1 <- factor(test_data1$feedback_type, levels = unique(c(test_data1$feedback_type, predicted_class1)))
actual2 <- factor(test_data2$feedback_type, levels = unique(c(test_data2$feedback_type, predicted_class2)))

predicted_class1 <- factor(predicted_class1, levels = levels(actual1))
predicted_class2 <- factor(predicted_class2, levels = levels(actual2))

# Calculate confusion matrices
confusionMatrix1 <- confusionMatrix(predicted_class1, actual1)
confusionMatrix2 <- confusionMatrix(predicted_class2, actual2)

# Print confusion matrices
print(confusionMatrix1)
print(confusionMatrix2)

# Calculate metrics
accuracy1 <- sum(actual1 == predicted_class1) / length(actual1)
accuracy2 <- sum(actual2 == predicted_class2) / length(actual2)

# Calculate Precision, Recall and F1 Score
precision1 <- confusionMatrix1$byClass['Pos Pred Value']
recall1 <- confusionMatrix1$byClass['Sensitivity']
f1Score1 <- 2 * (precision1 * recall1) / (precision1 + recall1)

precision2 <- confusionMatrix2$byClass['Pos Pred Value']
recall2 <- confusionMatrix2$byClass['Sensitivity']
f1Score2 <- 2 * (precision2 * recall2) / (precision2 + recall2)

# Calculate AUC
auc1 <- roc(actual1, predictions1)$auc
auc2 <- roc(actual2, predictions2)$auc

# Print the results
print(paste("Test1 - Accuracy:", accuracy1, "AUC:", auc1, "Precision:", precision1, "Recall:", recall1, "F1 Score:", f1Score1))
print(confusionMatrix1)
print(paste("Test2 - Accuracy:", accuracy2, "AUC:", auc2, "Precision:", precision2, "Recall:", recall2, "F1 Score:", f1Score2))
print(confusionMatrix2)

