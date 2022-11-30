# paritosh tripathi
# e20cse067

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score



def main():
    # Read the dataset
    df = pd.read_csv('voice.csv')

    # Check the shape of the dataset
    print("Shape of the dataset is ",df.shape, "\n")

    # Print the first 5 rows of the dataset
    print(df.head(), "\n")

    # Check the presence of missing values. Handle it if present 
    print(df.isnull().sum(), '\n')
    
    df.drop_duplicates(inplace=True)


    # Selecting the feature i.e., Identify the Independent variables and perform the extraction. 
    X = df.drop(['label'], axis=1)
    y = df['label']

    # Split the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Standardize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Apply PCA
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print("Variance ratio is",pca.explained_variance_ratio_, '\n')

    # Apply PCA with 0.95 variance
    pca = PCA(n_components=7)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    plt.scatter(X_train[:, 0], X_train[:, 1])
    plt.show()
    
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Apply Decision Tree and print the accuracy
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print("Confusion Matrix", "\n", confusion_matrix(y_test, y_pred), "\n")
    print("Accuracy is ",accuracy_score(y_test, y_pred), "\n")
    print("Precision is",precision_score(y_test, y_pred, average='macro'), "\n")
    print("Recall is ",recall_score(y_test, y_pred, average='macro'), "\n")

main()