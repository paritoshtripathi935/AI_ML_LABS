import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score

def main():
    import pandas as pd
    # ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']

    df = pd.read_csv("/home/paritosh/Semester 5/AI/AI_ML_LABS/lab_11/wine.data", header=None, columns=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline'])
    # 3. Check the presence of missing values. Handle it if present
    print(df.isnull().sum())
    
    # 4. Handle the categorical data if present using lable encoder
    #le = LabelEncoder()
    #df['Class'] = le.fit_transform(df['Class'])
    
    # 5. Selecting the feature i.e., Identify the Independent variables and perform the extraction.
    X = df.iloc[:, 1:14].values
    y = df.iloc[:, 0].values
    
    # 6. Select the n_component value (5)
    n_components = np.arange(1, 21)

    # 7. Training the GMM algorithm on the training dataset (Hint: from sklearn.mixture import GMM) (15)
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
    gmm.fit(X)
    
    # 8. Perform the predictioan and identify to which cluster data belongs (5)
    y_pred = gmm.predict(X)
    #print(y_pred)
    
    # 9. Training the GMM algorithm on the training dataset by setting (15) covariance_type="diag" covariance_type="full"
    gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=0)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    
    # 12. Check the accuracy of the model (5)
    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Confusion Matrix: ", confusion_matrix(y, y_pred))
    print("Classification Report: ", classification_report(y, y_pred))
    
    plt.scatter(X[:,0], X[:,1], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Malic acid')
    plt.show()
    
    plt.scatter(X[:,0], X[:,2], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Ash')
    plt.show()
    
    plt.scatter(X[:,0], X[:,3], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Alcalinity of ash')
    plt.show()
    
    plt.scatter(X[:,0], X[:,4], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Magnesium')
    plt.show()
    
    plt.scatter(X[:,0], X[:,5], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Total phenols')
    plt.show()
    
    plt.scatter(X[:,0], X[:,6], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Flavanoids')
    plt.show()
    
    plt.scatter(X[:,0], X[:,7], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Nonflavanoid phenols')
    plt.show()
    
    plt.scatter(X[:,0], X[:,8], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Proanthocyanins')
    plt.show()
    
    plt.scatter(X[:,0], X[:,9], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Color intensity')
    plt.show()
    
    plt.scatter(X[:,0], X[:,10], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Hue')
    plt.show()
    
    plt.scatter(X[:,0], X[:,11], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('OD280/OD315 of diluted wines')
    plt.show()
    
    plt.scatter(X[:,0], X [:,12], c=y_pred, s=40, cmap='viridis')
    plt.xlabel('Alcohol')
    plt.ylabel('Proline')
    plt.show()
    # calcualte sillohuette score and plot them
    silhouette_scores = []
    for n_cluster in n_components:
        gmm = GaussianMixture(n_components=n_cluster, random_state=0)
        gmm.fit(X)
        y_pred = gmm.predict(X)
        silhouette_scores.append(silhouette_score(X, y_pred))
    
    plt.plot(n_components, silhouette_scores)
    plt.xlabel("n_components")
    plt.ylabel("silhouette_score")
    plt.show()
