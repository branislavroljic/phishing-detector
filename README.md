# Phishing Email Detection
## Overview

This project is a comprehensive phishing email detection software designed to identify and classify phishing emails using various machine learning and deep learning models. The software implements models such as BERT, LSTM, Decision Tree, Random Forest, and others to explore different approaches to solving the phishing email detection problem.

## Folder Structure

  BERT/
  Contains the BERT model files and notebook:

        - bert.ipynb: BERT implementation and training.
        
        - bert_model.pt: Pretrained BERT model weights.
        
        - bert_optimizer.pt: Optimizer state for BERT.
        
  Same is for BERT Longformer
      

  LSTM/
    Contains the LSTM model files and notebook:
    
          -lstm.ipynb: LSTM implementation and training.
          
          -lstm_model.pt: Pretrained LSTM model weights.
          
          -lstm_optimizer.pt: Optimizer state for LSTM.
          
  Same if for Bi-LSTM and LSTM-Max-Pooling

  DecisionTree/
  Contains the Decision Tree implementation.

  GradientBoostingTree/
  Contains the Gradient Boosting Tree implementation.

  LogisticRegression/
  Contains the Logistic Regression implementation.

  NaiveBayes/
  Contains the Naive Bayes implementation.

  RandomForest/
  Contains the Random Forest implementation.

  data/
  Contains the dataset files:
  
      preprocessed_sub_and_body.csv: Preprocessed data for LSTM and BERT models, including email subjects and bodies.

  raw_data/
  Contains the original raw dataset files.

  app.ipynb
  This notebook handles preprocessing tasks, along with:\
      TF-IDF vectorization\
      Word2Vec embedding generation\
      Data preparation for various models

## Installation

    Clone this repository:

    git clone https://github.com/branislavroljic/nn-phish-detector.git
    cd nn-phish-detector

## Usage

  Run the preprocessing script in app.ipynb to generate TF-IDF, Word2Vec embeddings, and prepare the data for training.

  Train any of the implemented models by navigating to their respective folders and running the notebooks (bert.ipynb, lstm.ipynb, etc.).

  Pretrained models for BERT and LSTM can be loaded using bert_model.pt and lstm_model.pt.
