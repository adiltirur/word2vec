# word2vec Implementation in Tensorflow

## Word2vec implementation and Multiclass Document Classification with Saved embeddings from word2vec

### Required libraries
* Tensorflow
* Numpy
* tqdm
* re
* pandas
* sklearn
* matplotlib
* seaborn
### How to run the Program
* Change the working directory files from the word2vec.py , vector.py, domain_classification.py files to the respective directory in your local machine
* Point the dataset direction in the word2vec.py file
* Run word2vec.py file
* Now, point the metadata.tsv and embeddings.txt files (that is created during the execution of word2vec,py) in the vector.py
* Now make vectors for training,testing as well as development sets.
* Once you have all the 3 dataset(which will be a csv file) now first check the accuracy by running the domain_classification.py file.
* Before running it make sure that the training dataset is pointed correctly and instead of testing dataset give the dev dataset(so that we can check the accuracy)
* Once this is done, now point the testing set instead of dev dataset in domain_classification.py file.
* Run it and we will have the test_answers.tsv file generated with the file name and corresponding predicted labels.
