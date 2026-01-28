##############
# Clear environment
rm(list=ls())
gc()
options(warn=-1)  
##############

# Import libraries
.libPaths("/home/bagherir/R/x86_64-pc-linux-gnu-library/4.4")

library(dplyr)
library(ROCR)
library(logr)
library(matrixStats)
library(Matrix)
library(SGL)
library(pROC)
library(MLmetrics)
library(glmnet)
library(gglasso)
library(caTools)
library(caret)
library(tidyverse)
library(foreach)
library(doParallel)
library(grpreg)
library(trio)

##############

###############

start_time <- Sys.time()

formatted_datetime <- function() {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
}

#Enter the name of the phenotype to be Evaluated
phenotype_name = "TSLL"

cat("PFS for",phenotype_name,"at",formatted_datetime())

# Import data
cat("\nImport data",phenotype_name,"at",formatted_datetime())
Geno <- read.pedfile("genotype.ped")
char.pheno <- read.table("phenotypes.pheno", header = TRUE, stringsAsFactors = FALSE, sep = " ")
cat("\nImport data completed",phenotype_name,"at",formatted_datetime())

##############



##############
# Data Pre-processing
set.seed(100)
##############

cat("\nConvert the phenotype to matrix",phenotype_name,"at",formatted_datetime())

#Convert the phenotype to matrix
y <- matrix(char.pheno[,phenotype_name]) #Change the phenotype accordingly
rownames(y) <- char.pheno$IID #Change the feature according to the col name storing phenotype names
index <- !is.na(y)
y <- y[index, 1, drop = FALSE]

cat("\nCompleted to Convert the phenotype to matrix",phenotype_name,"at",formatted_datetime())


##############

cat("\nSelecting the indexed rows at",formatted_datetime())

Geno_y <- Geno[index, ]  # Only select the indexed rows

cat ("\nIndexed rows selected: ", formatted_datetime())

pheno_final <- data.frame(famid = rownames(y), y = y)

names(pheno_final)
Geno_y[1:5,1:10]

df <- merge(Geno_y, pheno_final, by = 'famid')

cat("\nCompleted Selecting the indexed rows at",formatted_datetime())

cat("\nSaving Geno_merged at",formatted_datetime())
write.csv(df,paste0("Geno_merged_",phenotype_name,"_df_prop.csv"))

##############

cat("\nPreparing df_final at ",formatted_datetime())

df_final <- df[, 7:dim(df)[[2]]] # Select the data set consisting of SNPs along with the phenotype
df_final <- sapply(df_final, as.character) #Keep SNP data as character
df_final <- df_final[sample(nrow(df_final)),] # Shuffling betweern the rows randomly
n <- dim(df_final)[[1]] # Get number of row value
d <- dim(df_final)[[2]] # Get number of column value

cat("\nCompleted preparing df_final at",formatted_datetime())

##############
#Determining the type of phenotype for Preliminary Feature Selection Test

determine_variable_type <- function(df, column) {
  # Filter out missing values
  non_missing_values <- df[[column]][!is.na(df[[column]])]
  
  # Determine if non-missing values are numeric
  if (is.numeric(non_missing_values)) {
    # Calculate the number of unique non-missing values
    unique_values <- length(unique(non_missing_values))
    total_values <- length(non_missing_values)
    unique_ratio <- unique_values / total_values
    # Determine if the variable is continuous or categorical
    if (unique_ratio > 0.05) {  # Adjust the threshold as needed
      return("Continuous")
    } else {
      return("Categorical")
    }
  } else {
    # If non-missing values are not numeric, assume categorical
    return("Categorical")
  }
}

#Stores the phenotype type into phenotype_type
phenotype_type <- determine_variable_type(char.pheno, phenotype_name)

##############

cat("\nPreliminary Feature screenning started at",formatted_datetime(),"\n")

# Preliminary Feature Screening
pvals <- rep(NA, d-1)

#############

# Would perform Anova test for fetching pvals if phenotype_type is Continuous or Chi Square Test if phenotype_type is Categorical
if (phenotype_type == "Continuous") {
  
  #Performing Anova test for Continuous Phenotype
  cat("\nAnova started")
  
  for(k in 1:d){
    if(length(unique(df_final[, k])) > 1) {
      pvals[k] <- summary(aov(df_final[, "y"] ~ df_final[, k]))[[1]][["Pr(>F)"]][1]
    } else {
      pvals[k] <- NA  # Assign NA for columns with only one one category
    }
  }
  
  cat("\nAnova ended")
  
} else if (phenotype_type == "Categorical") {
  
  #Performing Chi Square test for Categorical Phenotype
  cat("\n Chi Square Test Started")
  
  for(k in 1:d){
    tab = table(df_final[,k], df_final[,d])
    pvals[k] = chisq.test(tab)$p.value
  }
  
  cat("\n Chi Square Test Ended")
}

##########

cat("\nCompleted Preliminary Feature screenning at",formatted_datetime(),"\n")

##############

# Finding quantiles
cat("\n Quantile pvals \n ")
quantile_pvals = quantile(pvals, na.rm = TRUE)
print(quantile_pvals)

##############

cat("\n # Identify ones that passed the filter",formatted_datetime())

# Identify ones that passed the filter.
a1 <- which(pvals < 0.10)
a2 <- which(pvals < 0.05)
a3 <- which(pvals < 0.01)
a4 <- which(pvals<0.001) #1 in thousand
a5 <- which(pvals<0.0001) #1 in 10 thousand
a6 <- which(pvals<1e-5)
a7 <- which(pvals<1e-6)
a8 <- which(pvals<1e-7)
a9 <- which(pvals<1e-8)
a10 <- which(pvals<1e-9)
a11 <- which(pvals<1e-10)
a12 <- which(pvals<1e-17)

cat("\nCompleted identifying pvals filtered completed",formatted_datetime(),"\n")

#Print the number of filtered snps according to the significance level
cat("Indices with pvals < 0.10:", length(a1), "\n")
cat("Indices with pvals < 0.05:", length(a2), "\n")
cat("Indices with pvals < 0.01:", length(a3), "\n")
cat("Indices with pvals < 0.001:", length(a4), "\n")
cat("Indices with pvals < 0.0001:", length(a5), "\n")
cat("Indices with pvals < 1e-5:", length(a6), "\n")
cat("Indices with pvals < 1e-6:", length(a7), "\n")
cat("Indices with pvals < 1e-17:", length(a8), "\n")
cat("Indices with pvals < 1e-17:", length(a9), "\n")
cat("Indices with pvals < 1e-17:", length(a10), "\n")
cat("Indices with pvals < 1e-17:", length(a11), "\n")
cat("Indices with pvals < 1e-17:", length(a12), "\n")


##############
cat("\nCreate a new data set for filtered SNPs at",formatted_datetime())

# Create a new data set for filtered SNPs
col_name <- list()

# Initialize an empty data frame
df_new <- matrix(nrow=n,ncol = 0)
df_new <- data.frame(df_new)

cat("\nSelecting a5 as the significance level.\n")
for (i in a5) {
  df_new = df_new %>% add_column(df_final[,i])
  col_name = append(col_name,colnames(df_final)[i])
}

colnames(df_new) <- col_name
d1 <- dim(df_new)[[2]]

cat("\nCreated a new data set for filtered SNPs at",formatted_datetime())

#############
# Encoding the dataset to numeric values saving the dataset 

dim_df_new <- dim(df_new)

cat("\nDimensions of encoded data would be:", dim_df_new)

cat("\nEncoding the Data at", formatted_datetime())
#Encoding the Data of ped file
df_new[df_new == 'A'] <- 0  # Converting A to 0 
cat("\nA completed at:", formatted_datetime())
df_new[df_new == 'T'] <- 1  # Converting T to 1
cat("\nT completed at:", formatted_datetime())
df_new[df_new == 'G'] <- 2  # Converting G to 2
cat("\nG completed at:", formatted_datetime())
df_new[df_new == 'C'] <- 3  # Converting C to 3
cat("\nC completed at:", formatted_datetime())


#Converting it to the numeric data
df_new <- sapply(df_new,as.numeric)
df_new <- data.frame(df_new)

# Save the filtered data set
write.csv(df_new, paste0(phenotype_name,"_filtered_prop.csv"))
cat("\n Saved the filtered data",formatted_datetime())


##############
#Splitting data in 50% train and 50% test sets
cat("\n Splitting data started at:", formatted_datetime())


# Get the number of observations (rows) in the data frame df_new
nobs <- nrow(df_new)

# Create 5-fold cross-validation indices based on the row means of df_new
id <- createFolds(rowMeans(df_new), k=5, list=FALSE)

# Randomly sample 50% of the observations to be in the training set
training.id <- sample(seq_len(nobs), size = 0.5 * nobs)

# The test set is the remaining observations not in the training set
testData <- df_new[-training.id, ]

# The training set is the randomly sampled 50% of the observations
trainData <- df_new[training.id, ]

# Convert the training data to a data frame (though it already is one)
data.train <- as.data.frame(trainData)

# Convert the test data to a data frame (though it already is one)
data.test <- as.data.frame(testData)

##############

#####
#Save the train and test sets for further analysis of Neural Networks
x_test <- as.matrix(data.test[,1:(d1-1)])
write.csv(x_test, paste0("XTest_", phenotype_name, "_prop.csv"))

y_test <- as.matrix(data.test[, d1])
write.csv(y_test, paste0("ytest_", phenotype_name, "_prop.csv"))

x_train <- as.matrix(data.train[,1:(d1-1)])
write.csv(x_train, paste0("XTrain_", phenotype_name, "_prop.csv"))

y_train <- as.matrix(data.train[, d1])
write.csv(y_train, paste0("ytrain_", phenotype_name, "_prop.csv"))

cat("\n Splitting data completed at:", formatted_datetime())

############

#Noting down the total runtime 
end_time <- Sys.time()

# Calculate the total run time
run_time <- end_time - start_time

# Print the start time, end time, and total run time
cat("\nStart time:", start_time, "\n")
cat("\nEnd time:", end_time, "\n")
cat("\nTotal run time:", run_time, "\n")
#############
