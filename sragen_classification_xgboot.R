#Credentials
Sys.setenv("AWS_ACCESS_KEY_ID"     = "YOUR-AWS-ACCESS-KEY", "AWS_SECRET_ACCESS_KEY" = "YOUR-AWS-SECRET-ACCESS-KEY", "AWS_DEFAULT_REGION"    = "ap-southeast-1", "AWS_ENDPOINT"          = "ec2-southeast-1.amazonaws.com")

# Load libraries
library(sits)
library(gdalcubes)
library(reticulate)
library(tensorflow)
library(tibble)
library(magrittr)
library(profvis)
library(randomForest)

# Region of interest
roi <- c("xmin" = 474578.8591,
          "xmax" = 519171.3366,
          "ymin" =  9167194.1834,
          "ymax" = 9197895.9217) 


# Search Sentinel-2
s2_cube <- sits_cube(
  source = "MSPC",
  collection = "SENTINEL-2-L2A",
  tiles = c("49MDM", "49MEM"),
  bands = c(  "B02","B03","B04","B05", "B08","B11","CLOUD"),
  start_date = as.Date("2021-01-01"),
  end_date = as.Date("2021-07-01")
)

# Regularization
# creating an regular data cube from MSPC
s2_cube_regular <- sits_regularize(
              cube = s2_cube, 
              output_dir  = "./added/", 
              period = "P15D", # 15 days period
              res = 60, #60-meter resolution to make the process faster
              agg_method = "median", 
              multicores = 8)

# Load CSV
samples <- read.csv("sampel3.csv")

# Training samples
samples_S2_49MDM_MEM_2021_regular <- sits_get_data(s2_cube_regular, samples = samples, multicores=8)


# Train with 6 bands
samples_s2_6bands <- sits_select(samples_S2_49MDM_MEM_2021_regular,
                                 bands = c("B02","B03","B04", "B05","B08", "B11"))

# build the classification model
# 6 Bands
xgb_model6 <- sits_train(
  data      = samples_s2_6bands,
  ml_method = sits_xgboost()
) 

# classify the cube using an xgboost model
s2_probs_6bands <- sits_classify(
  s2_cube_regular,
  xgb_model6,
  memsize = 24,
  multicores = 8,
  output_dir  = "./images_6bands/", verbose=TRUE
)

# plot the probabilities
plot(s2_probs_6bands)

# Bayesian Smoothing
s2_bayes6 <- sits_smooth(s2_probs_6bands, type = "bayes", output_dir = "./images_6bands/")

# Plot
plot(s2_bayes6)

# Label
s2_label6 <- sits_label_classification(s2_bayes6, output_dir = "./images_6bands/")

#validation
val_xgboost <- sits_kfold_validate(samples_s2_6bands, 
                                folds = 5, 
                                ml_method = sits_xgboost())
# Show the validation result
val_xgboost
