## A Deeper Look at Computational Sarcasm

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
 2. ResMet baseline
 3. DweNet baseline
 
Each folder contains the source code of the final models, as well as a jupyter notebook for running the models.

#### Results

The results of each experiment, as well as data used to analyse findings are contained within 'Sarcasm/Results/'.
