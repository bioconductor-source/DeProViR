## ----echo=FALSE,message=FALSE, warning=FALSE----------------------------------
library("knitr")

## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ---- message=FALSE, warning=FALSE, eval=TRUE---------------------------------

options(timeout=240)
library(tensorflow)
library(data.table)
library(DeProViR)

tensorflow::set_random_seed(101)
model_training <- modelTraining(
      url_path = "https://nlp.stanford.edu/data",
      training_dir = system.file("extdata", "training_Set",
                              package = "DeProViR"),
      input_dim = 20,
      output_dim = 100,
      filters_layer1CNN = 32,
      kernel_size_layer1CNN = 16,
      filters_layer2CNN = 64,
      kernel_size_layer2CNN = 7,
      pool_size = 30,
      layer_lstm = 64,
      units = 8,
      metrics = "AUC",
      cv_fold = 2,
      epochs = 100,
      batch_size = 128,
      plots = FALSE,
      tpath = tempdir(),
      save_model_weights = FALSE,
      filepath = tempdir()) 

## ----message=FALSE, warning=FALSE---------------------------------------------
options(timeout=240)
library(tensorflow)
library(data.table)
library(DeProViR)
pre_trainedmodel <- 
   loadPreTrainedModel()


## -----------------------------------------------------------------------------
#load the demo test set (unknown interactions)
testing_set <- fread(
   system.file("extdata", "test_Set", "test_set_unknownInteraction.csv",
                                           package = "DeProViR"))
scoredPPIs <- predInteractions( 
    url_path = "https://nlp.stanford.edu/data",
                 testing_set,
                 trainedModel = pre_trainedmodel)
scoredPPIs


## ----warning=FALSE, message=FALSE, eval=TRUE----------------------------------
# or using the newly trained model 
predInteractions(url_path = "https://nlp.stanford.edu/data",
                 testing_set,
                 trainedModel = model_training)


## ---- eval=TRUE---------------------------------------------------------------
sessionInfo()

