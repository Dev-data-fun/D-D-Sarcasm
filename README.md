## A Deeper Look at Computational Sarcasm

### Local Setup

This work was done on an Ubuntu 18.04.3 system using a GeForce RTX2080 GPU for processing. 

The following package versions were used: Python V3.7.1, Pytorch V1.0.1, Fastai V1.0.50, Cuda V10.1.105 and CuDNN V7.5.0
                                          

### Data, Models and Results

#### Datasets

The three datasets are located under 'Sarcasm/Datasets/', they are:

  1. Headlines
  2. Reddit Main
  3. Reddit Pol

Within each subfolder is an 'Embeddings' folder containing the weight matrix corresponding to the word embeddings for each respective dataset. The 4 available word embeddings are: 

 1. 50D GloVe embeddings
 2. 300D FastText1M embeddings
 3. 300D FastText1M subword embeddings
 4. 300D FastText2M embeddings
 
#### Models

The three models implemented in our work are located under 'Sarcasm/Models/', they are:

 1. CNN baseline
 2. ResNet baseline
 3. DweNet baseline
 
Each folder contains the source code of the final models, as well as a jupyter notebook for running each model.

#### Results

The results of each experiment, as well as data used to analyse findings are contained within 'Sarcasm/Results/'.
