## A Deeper Look at Computational Sarcasm

### Local Setup

This work was done on an Ubuntu 18.04.3 system using a GeForce RTX2080 GPU for processing. 

The following package versions were used: Python V3.7.1, Pytorch V1.0.1, Fastai V1.0.50, Cuda V10.1.105 and CuDNN V7.5.0.
                                          
### Project Structure

The complete project is 324MB.


```bash
└── Sarcasm
    ├── Datasets
    │   ├── Headlines
    │   │   ├── Embeddings
    │   │   └── Headlines.csv
    │   ├── Reddit Main
    │   │   ├── Embeddings
    │   │   ├── Parent
    │   │   │   └── Embeddings
    │   │   └── RedditMain.csv
    │   └── Reddit Pol
    │       ├── Embeddings
    │       └── RedditPol.csv
    ├── Models
    │   ├── CNN_baseline
    │   ├── DweNet
    │   └── ResNet_baseline
    └── Results
        ├── Benchmarks
        │   ├── Case Study
        │   ├── Heatmap
        │   └── Model Benchmarks
        │       ├── Baseline CNN
        │       ├── Baseline ResNet
        │       └── DweNet
        ├── Data Preprocessing Investigation
        ├── Depth Investigation
        ├── Embedding Investigation
        ├── Growth Investigation
        └── Local Context Investigation
```

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
 
Each folder contains the source code of the final models, as well as a jupyter notebooks (.pynb files) for loading, training and testing the models. Performance is presented in accuracy and F1 score. 

#### Results

The results of each experiment, as well as data used to analyse findings are contained within 'Sarcasm/Results/'.

The experiments performed were:

 1. Benchmarking - Performance of each model, on each dataset
 2. Data Preprocessing - Effect of different data preprocessing
 3. Depth Investigation - Effect of differing depths
 4. Embeddings Investigation - Effect of different embeddings, using static and non-static representations
 5. Growth Investigation - Effect of the growth rate in DweNet
 6. Local Context Investigation - Effect of local context 
 
### Running the Experiments

Instructions for runnning benchmarks and exploratory investigations in this work:

#### Benchmarks

To run the benchmark, open the jupyter notebook corresponding to the model you wish to run. 7 variables must be set:

 1. PADDING - 64 for Headlines dataset, 128 for Reddit Main and Reddit Pol
 2. DATASET_PATH - Path to folder containing data: '.../Datasets/Headlines/' for Headlines folder.
 3. DATASET - Dataset file name: 'Headlines.csv' for Headlines folder.
 4. COL - Column to load data from csv: 'headline' for Headlines, 'comment' for Reddit Main.Pol
 5. WEIGHTS - Path to word embeddings: '.../Datasets/Headlines/Embeddings/glove/Weights_glove_headlines.pkl' for GloVE embeddings for Headlines data
 6. EMBED_DIM - Dimension of word embedding: 50 for GloVe, 300 for FastText
 7. STATIC - Static or Non-static embeddings: Non-static leads to better performance
 
 Run the code cells in the notebook to run the dataset with the chosen embedding. Results will be displayed in terms of accuracy and F1 score.
 
 #### Data Preprocessing
 
 To compare benchmark techniques use the embeddings located under 'Sarcasm/Results/Data Preprocessing Investigation/'. The following 2 options are available:
 
  1. Min 1 - Headlines word embedding for a word threshold of 1
  2. Min 2 - Headlines word embedding for a word threshold of 1
  
 **Note:** Only the Headlines dataset is available for this investigation. To use the default preprocessing simply load the standard Headlines embedding.
 
 The following change will need to be made to the jupyter notebook:
 
  1. Edit code cell 5 as such: 
  ```python processor = [TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(min_freq=X)]```
     where X is the Min frequency embedding chosen
 
 To evalute the performance of the pre and postprocessing rules, make the following change to the jupyter notebook:
  
  1. Edit code cell 5 as such: 
  ```python tokenizer = Tokenizer(SpacyTokenizer, 'en', pre_rules=[], post_rules=[])```
 
 #### Embedding Investigation
 
 To compare the different word embeddings available, select the jupyter notebook corresponding to the model you wish to use. The 3 FastText embeddings are available under the Embeddings folder for each dataset. 
 
 Make sure to set the EMBED_DIM variable to 50 for GloVe, and 300 for FastText.
 
 To use compare static vs non-static representations, set the STATIC variable accordingly.
 
 #### Local Context Investigation
 
 To evaluate the effect of local context - provided in the form of parent comments on the Reddit Main dataset - set WEIGHTS to the Parent word embedding for the Reddit Main dataset located under: 'Sarcasm/Datasets/Reddit Main/Parent/Embeddings/glove/'. Then make the following changes to the jupyter notebook:
 
 1. Edit code cell 5 as such: 
 ```python processor = [TokenizeProcessor(tokenizer=tokenizer, mark_fields=True), NumericalizeProcessor()]```
 2. Edit code cell 6 as such: 
 ```python data = (TextList.from_csv(DATASET_PATH, DATASET, cols=[COL, 'parent'], processor=processor))``` 
    This will load the parent comment and concatenate it to the child comment, seperated by the 'xxfld' token.
 
**Note:** Only the Reddit Main dataset is available for this investigation.


### Potential Issues

If the model returns the following error: ```RuntimeError: cuda runtime error (59)```, it is likely caused by words in the vocabulary that are not present in the embedding (this can result from using a newer version of fastai). To fix this, save the vocab of the data once the learner has been instantiated as such:

```python pickle.dump(data.vocab.itos, FILE.pkl)```

Then download the desired embedding from its respective site, and build the weight matrix using the following:

```python 

itos = PATH   #path to itos vocab file saved above
weights_matrix = np.zeros((len(itos), EMBED_DIM))
words_found = 0

def gather(emb_dict):

    global words_found
    global weights_matrix
    global unknown_words

    for i, word in tqdm(enumerate(itos)):
        if i==1:
            continue       # We skip the <pad> token, leaving it to be zeros
        try:
            weights_matrix[i] = list(emb_dict[word])
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBED_DIM,))
```

Then pass the embedding file to the 'gather' function and save the resulting 'weights_matrix' object.
