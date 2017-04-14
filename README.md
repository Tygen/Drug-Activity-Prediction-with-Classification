# Drug Activity Prediction

	Approach 1: Random Forest algorithm
F1: 0.6875
Libraries  used: numpy, scipy, ensemble

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:	
Read files train.dat, test.dat: we create a training_list and test_list
After that calculation for featureList is done: intersection of unique words from linesOfTrainData and unique words from linesOfTestData is considered for this.
Now we calculated the CSR matrix from both  training_list and test_list based on feature list.
L2 norms are calculated for the CSR matrices
The RandomForestClassifier is used from sklearn.ensemble  to computed the prediction
 For given number of n_estimators the RandomForestClassifier will create that many number of trees
After feeding training_matrix and class_list for learning.
And prediction is written on format.dat






	Approach 2: NeuralNet algorithm.
F1: 0.5600
Libraries  used: numpy, scipy, neural_network

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:
Read files train.dat, test.dat: we create a training_list and test_list.
After that calculation for featureList is done: intersection of unique words from linesOfTrainData and unique words from linesOfTestData is considered for this.
Now we calculated the CSR matrix from both  training_list and test_list based on feature list.
L2 norms are calculated for the CSR matrices
The MLPClassifier was used from sklearn.neural_network  to computed the prediction
 For given solver='lbfgs', alpha=1e-5 and hidden_layer_sizes=(50, 20)in the MLPClassifier. 
After feeding training_matrix and class_list for learning.
And prediction is written on format.dat







	Approach 3: BernoulliNB algorithm
F1: 0.4348
Libraries  used: numpy, scipy, naive_bayes

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:	
Read files train.dat, test.dat: we create a training_list and test_list
After that calculation for featureList is done: intersection of unique words from linesOfTrainData and unique words from linesOfTestData is considered for this.
Now we calculated the CSR matrix from both  training_list and test_list based on feature list.
L2 norms are calculated for the CSR matrices
The BernoulliNBClassifier is used from sklearn.naive_bayes  to computed the prediction
For given number of n_estimators the BernoulliNBClassifier will create that many number of trees.
After feeding training_matrix and class_list for learning.
And prediction is written on format.dat








	Approach 4: KNN algorithm
F1: 0.6667
Libraries  used: numpy, scipy

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:	
Read files train.dat, test.dat: we create a linesOfTrainData and linesOfTestData.
Initialize CountVectorizer
Fit_transform linesOfTrainData
Transform linesOfTestData
cosineSimilarityValue is calculated with, function cosine_similarity(vt,vs).
 For given number of K, we identified the nearest neighbours.
And for those index values we checked if we have +1 or -1 in given list of training data.
And average of max(count(+1), count(-1)) is written on format.dat










	Approach 5: SVM algorithm
F1: 0.3636
Libraries  used: numpy, scipy, sklearn

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:	
Read files train.dat, test.dat: we create a training_list and test_list
After that calculation for featureList is done: intersection of unique words from linesOfTrainData and unique words from linesOfTestData is considered for this.
Now we calculated the CSR matrix from both  training_list and test_list based on feature list.
L2 norms are calculated for the CSR matrices
The SVC Classifier is used from sklearn.svm to computed the
prediction.
For given number of c=0.1 the SVC is calculated.
After feeding training_matrix and class_list for learning.
And prediction is written on format.dat









	Approach 6: Perceptron algorithm
F1: 0.7879
Libraries  used: numpy, scipy, linear_model

Number of features considered: 91598
Training data:  [nrows 800]
 	Test data: [nrows 350]

#### Steps:	
Read files train.dat, test.dat: we create a training_list and test_list
After that calculation for featureList is done: intersection of unique words from linesOfTrainData and unique words from linesOfTestData is considered for this.
Now we calculated the CSR matrix from both  training_list and test_list based on feature list.
L2 norms are calculated for the CSR matrices
The Perceptron Classifier is used from sklearn.linear_model  to computed the prediction
 After feeding training_matrix and class_list for learning.
And prediction is written on format.dat
