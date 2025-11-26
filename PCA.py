# The following code is an implementation of Principal Component Analysis (PCA) from scratch for a three dimensional dataset.
# Our aim is to reduce the dimensionality of the dataset from 3D to 2D while retaining most of the variance in the data.
# Dataset description : The dataset consists of three features (X1, X2, X3) and a target variable (target). 
# It is not importanat what exactly the features represent. here we are focusing on the working of PCA.
# You can extraploate this code to higher dimensions as well.

import numpy as np
import pandas as pd

def pca(df):

    # Apply standard scaling aka mean centering : 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    # Find the covariance matrix :
    cov_matrix = np.cov(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])

    # Find the eigen values and eigen vectors :
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

    # Selecting the top 2 eigen values and corresponding eigen vectors :
    pc = eigen_vectors[0:2]

    # Projecting the data on the new feature space :
    transformed_df = df.iloc[:, :-1].dot(pc.T)

    # Generating the final dataframe with the target variable :
    new_df = pd.DataFrame(transformed_df,columns=['PC1','PC2'])
    new_df['target'] = df.iloc['target'].values


    return new_df
