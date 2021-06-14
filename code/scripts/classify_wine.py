from typing import Iterator, Iterable, Tuple, Text, Union
import re
import numpy as np
import pandas as pd
from numpy import array
from scipy.sparse import spmatrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, accuracy_score

import seaborn as sns 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize

stopwords = set(stopwords.words('english'))




NDArray = Union[np.ndarray, spmatrix]
       
def DataToSeries(path: str, feature_cols: str, target_cols: str, test_size, random_state):
    
    """  
    Initializes an object for converting textfile to training dataframe (pandas series).
    :param path: The path of a wine review file.
    :param feature_cols: The name of the columns using to create features to fit the model later
    :param target_cols: The name of the column using as a target variable.
    :param test_size: The proportion between the test set size and the total size of the dataset
    :param random_state: random state in shuffling
    :return: Pandas series for features and target variables for training set and validating/test set.
    """
    data = pd.read_csv(path, index_col = False)
    size = data.shape[0] 
    feature_train, feature_test, target_train, target_test=train_test_split(data[feature_cols],data[target_cols],test_size = test_size,     random_state = random_state)
    return feature_train, feature_test, target_train, target_test
   
def preprocess_features(feature_series):       
    """  
        The feature series is/are preprocessed by lowering letters, removing stop words and non-alphabetical characters.
        :param feature_series: The feature series/columns of our wine data.
        :return: Pandas series for features and target variables (values 0,1,2,3,4) after preprocessing.
        """
    detokenizer = TreebankWordDetokenizer()
    def clean_description(desc):
        desc = word_tokenize(desc.lower())
        desc = [token for token in desc if token not in stopwords and token.isalpha()]
        return detokenizer.detokenize(desc)
    
    preprocessed_feature_series = feature_series.apply(clean_description)
    size = preprocessed_feature_series.shape[0]
    return preprocessed_feature_series  

def preprocess_targets(target_series):
    """  
        The target series is the column Points of the Wine dataframe. The points are in the scales 80-100.
        We use a specific scale to categorize the target series into five classses.
        :param target_series: The rating point of users for the wines.
        :return: Pandas series for features and target variables (values 0,1,2,3,4) after preprocessing.
        """
    def points_to_class(points):
        if points in range(80,83):
            return 0
        elif points in range(83,87):
            return 1
        elif points in range(87,90):
            return 2
        elif points in range(90,94):
            return 3
        else:
            return 4
    preprocessed_target_series = target_series.apply(points_to_class)
    return preprocessed_target_series

class TextToFeatures:
    def __init__(self, texts: Iterable[Text], ngram_range: Tuple[int, int]):
        """  
        Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. 

        :param texts: The training texts.
        :ngram: tuples of int to select n-grams
        """
        
        self._vec = CountVectorizer(ngram_range=ngram_range,analyzer='word',stop_words=stopwords,token_pattern=r'\w{1,}')
        self._vocab = self._vec.fit_transform(texts)

   

    def __call__(self, texts: Iterable[Text]):
        """
        Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        self._features = self._vec.transform(texts)
        return self._features

        
class Classifier:
    def __init__(self, cv, random_state, class_weight, solver, max_iter = 7600):
        """
        Initalizes a logistic regression classifier.
        :cv: number of folds in cross validation
        """
        
        self._logreg = LogisticRegressionCV(cv=cv,random_state=random_state,class_weight=class_weight,                            multi_class='multinomial',solver=solver,max_iter=max_iter)

    def train(self, features, classes):
        """
        Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param classes: A class vector, where each entry represents a class.
        """
        
        self._clf = self._logreg.fit(features, classes)
    
    def predict(self, features):
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
 
        return self._clf.predict(features)