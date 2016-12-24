# Congressional Roll Call Vote Prediction Using Neural Nets

This repo contains a few attempts at predicting congressional roll call votes using neural networks. All the code is based on tensorflow.

So far I've implemented three models:

1. Softmax regression
2. 1-Layer deep network
3. 1-Layer recurrent network

##Data
I tested this on the 112th congress only so far. Inputs to the models are congressman-level features plus unigram counts for the text of bills being voted on. The validation and test datasets contain only votes and bills that occurred **after** the votes in the training data. This is to maintain temporal consistency. 
All categorical features are one-hot encoded, while numerical ones are standardized before being fed to the model. Word counts are replaced with tf-idf term frequency. 

|Data|Observations|Different Bills|Congressman Features|Vocabulary Length|
|---|----|---|---|---|
|Train|90795|267|88|9970|
|Valid|2003|8|88|9970|
|Test|937|8|88|9970|

## Network Architectures 

### Word embeddings


The words are fed into the embedding layer, which reduces the dimensionality of the words to 50. To do so, each word is represented by a vector of dimension 50 in the layer. Each word frequency scales its corresponding embedded vector and then all the vectors are added together. This implies that each bill is a weighted average of the embedding vectors corresponding to the words in it. 

As an example, let's say a bill contains the word "roads" with a tf-idf frequency of 0.15. As the word is fed to the embedding layer, its relative vector is looked up in the embedding matrix, which contains a length-50 vector for each of the 9970 words in the vocabulary. Then this vector is scaled by 0.15 and added together with all the other scaled vectors of words in the bill. 

** The values of the embedding matrix are optimally chosen by the network during training** This implies that the embeddings are optimized for the prediction tasks together with the weights. This is powerful feature of neural networks, as the dimensionality reduction is made with the prediction task as objective. So the resulting dimensions will be the most informative for the task at hand. This improves on other dimensionality reduction methods such as SVD or LDA. 

### Deep Net:

This is a simple deep network with 1 hidden sigmoid layer with dimension 36. Added to this there is an embedding layer of size 50 that reduces the dimensionality of the vocabulary before going to prediction. This implies that the hidden layer gets fed 88 + 50 features instead of 88 + 9970. 

The networks' trainable parameters are: 

1. An embedding matrix of dimension 9970 x 50
2. A weight matrix **W** with dimension 88+50 x 36
3. A bias vector **b** with dimension 36.
4. A weight matrix **U** with dimension 36 x 2
5. A bias vector **a** with dimension 2. 

The network works as follows: 

1. Each observation is split into two vectors, one of 88 features and another one of 9970 word counts for the bill. 
2. The word vector is fed to the embedding layer, which outputs a representation vector of length 50. 
3. The two vectors are combined into a feature vector of length 88 + 50, call it **x**
4. This vector is fed to the hidden layer that outputs a vector **h** according to the following equation:
[$$\mathbf{h} = Sigmoid(\mathbf{xW} + \mathbf{b})$$]])
5. **h** is fed into a projection layer that outputs a probability vector with one entry for each class (either 1 or 0)
$$\mathbf{\hat{y}} = Softmax(\mathbf{hU} + \mathbf{a})$$
6. The class with the highest probability is chosen as the network's prediction. 

### Recurrent Net: 

The recurrent net is also very simple. There is still an embedding layer to reduce feature dimensionality, plus one recurrent hidden layer. This layer gets fed its previous state as well as its current input to make predictions. The sequence length I use for each training step is 4, I.E. at each training step the network will update its state on 4 different bills in sequence. 

The main idea behind using this network is that we should take advantage of the voting history of congress, I.E: how previous bills were voted on is likely to be a good predictor of how following bills will be voted on. 

This is still a very rough implementation of this concept, as all votes by all congressmen are weighted equally. 

The networks' trainable parameters are: 

1. An embedding matrix of dimension 9970 x 50
2. A weight matrix **W** with dimension 88+50 x 36
3. A Weight matrix **I** with dimension 36 x 36
4. A bias vector **b** with dimension 36.
5. A weight matrix **U** with dimension 36 x 2
6. A bias vector **a** with dimension 2. 

The network works as follows: 

1. Each observation (t) is split into two vectors, one of 88 features and another one of 9970 word counts for the bill. 
2. The word vector is fed to the embedding layer, which outputs a representation vector of length 50. 
3. The two vectors are combined into a feature vector of length 88 + 50, call it **x_t**
4. This vector is fed to the hidden layer together with the output from the previous observation (state) 
$$\mathbf{h}_t = Sigmoid( \mathbf{h}_{t-1} \mathbf{I} + \mathbf{x_tW} + \mathbf{b})$$
5. **h** is fed into a projection layer that outputs a probability vector with one entry for each class (either 1 or 0)
$$\mathbf{\hat{y}} = Softmax(\mathbf{h_tU} + \mathbf{a})$$
6. The class with the highest probability is chosen as the network's prediction. 

## Main Results
The main result for all the models is the same: **All the models end up labelling all the points as 1 (yay).**

### Why is this happening?
Because the data is **extremely** skewed. Here are proportions for our data on the 112th congress:

|Data|Yay|Nay|
|----|----|----|
|Train|0.964|0.036|
|Valid|0.967|0.033|
|Test|0.950|0.050|

All the models start out labelling everything as 0 because I weight the observations so that 0s matter more. After a few epochs in training they realize that they need to label everything as 1 if they want to win and they do. 

### Have you tried oversampling?
Yes,  but for the **deep network** only. I oversampled training data using the [SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf) algorithm and inserted synthetic observations in the dataset until the proportion of 0s was about 20%. The deep net performs better in training in terms of recall, however it still does poorly on the validation set. This leads me to believe that it is somehow overfitting the training data, although all weights have an associated regularisation term in the loss function. 

I'm not sure if oversampling is a great solution in general, we get fewer true positives and our accuracy goes down. 


## TODO:

* Implement a better prediction framework for all the models.
* Try LSTM gates on the RNN
* Play around with DN architecture
* Implement oversampling for RNN
* Add 111th and 113th to training data. 
*  Write a function to take predictions and plot ROC curve, also calculate AUC. 

