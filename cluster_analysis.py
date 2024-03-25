import pandas as pd
import numpy as np

# cluster detection
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import nltk

import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn as sns #Makes the graphics look nicer

from ast import literal_eval

def get_data(col='normalized_tokens', filename='~/Study/MACSS/macs30200/data/comments_df.csv'):
    '''
    Get csv file and import as Dataframe

    - col: column of tokens in dataframe
    - filename: file path of csv file
    '''
    df = pd.read_csv(filename)

    # if column is str type, need literal_eval to turn it back into a list
    if isinstance(df[col][0], str):

        def converter(x):
            return literal_eval(x)
        
        # may need change so that we are not calling columns of df with native name
        df = pd.read_csv(filename, converters={'tokens': converter, 'normalized_tokens': converter})

    return df

def bigrams_freq(df, text_col, n):
    '''
    Get top-n bigrams of a column of tokens in dataframe
    
    - df: Dataframe
    - text_col: column of tokens to get bigrams from
    - n: number of most frequent bigrams
    '''
    
    # intialize bigram finder
    bigrams = nltk.collocations.BigramCollocationFinder.from_words(df[text_col].sum())
    
    # scoring method
    def bigramCount(count, wordsTuple, total):
        return count
    
    scores = bigrams.score_ngrams(bigramCount)

    return scores[:n]

    
def plotSilhouette(df, col1, col2, n_clusters, n_words):
    '''
    Helper function to calculate average silhouette score for sample and plot silhouette and clustered data
    and display most distinguishing words for each cluster
    - df: dataframe
    - col1: column of input tokens
    - col2: column of tokens changed from list to str type
    - n_clusters: number of clusters
    - n_words: number of distinguishing words
    '''
    
    # change column of tokens from list to string
    df[col2] = df[col1].apply(lambda x: ' '.join(item for item in x))

    # intialize tdidf vectorizer
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=3, max_features=1000, stop_words='english', norm='l2')
    tdidf_matrix = vectorizer.fit_transform(df[col2])

    X = tdidf_matrix.toarray()

    # reduce dimensionality to 2 principal components
    PCA = sklearn.decomposition.PCA
    pca = PCA(n_components=2).fit(X)
    reduced_X = pca.transform(X)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

    #ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10, init='k-means++')
    cluster_labels = clusterer.fit_predict(X)

    # add column of predicted labels to df
    df['kmeans_pred'] = cluster_labels

    # compute mean silhouette scores/coefficient over all samples
    silh_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

    # Compute silhouette coefficient for each sample
    sample_silh_values = sklearn.metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        # get silh scores for cluster
        ith_cluster_silh_values = sample_silh_values[cluster_labels==i]
        ith_cluster_silh_values.sort()

        cluster_size = ith_cluster_silh_values.shape[0]
        y_upper = y_lower + cluster_size
        cmap = matplotlib.cm.get_cmap('nipy_spectral')
        color = cmap(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper), 0, ith_cluster_silh_values, facecolor=color, 
                          edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i))

        y_lower = y_upper + 10
    
    ax1.set_title('Silhoutte plot for each cluster')
    ax1.set_xlabel('Silhouette score')
    ax1.set_ylabel('Cluster label')

    ax1.axvline(x=silh_avg, color='red', linestyle='--')

    ax1.set_yticks([]) # clear y-axis label and ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = cmap(float(i) / n_clusters)
    ax2.scatter(reduced_X[:, 0], reduced_X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # label clusters
    centers = clusterer.cluster_centers_
    projected_centers = pca.transform(centers)
    # draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1], marker='o', c='white', alpha=1, s=200)
    # write in cluster number in the centers
    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=30)
    
    ax2.set_title('Visual of clustered data')
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')

    plt.suptitle(f'Silhouette analysis for KMeans clustering with n_clusters={n_clusters}', fontsize=14, fontweight='bold')
    plt.show()

    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silh_avg}")

    # Most distinguishing terms per cluster
    print('Most-distinguishing terms per cluster:')

    # terms/words
    terms = vectorizer.get_feature_names()

    # re-order terms
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

    for i in range(n_clusters):
        print(f'Cluster {i}:')
        for ind in order_centroids[i, :n_words]:
            print(f'{terms[ind]}')
        print('\n')

    return cluster_labels

if __name__ == "__main__":

    col = 'normalized_tokens'
    df = get_data(filename='~/Study/MACSS/macs30200/data/comments_df.csv')
    n_bigrams = 25
    bigrams = bigrams_freq(df, col, n_bigrams)
    print(f'Top {n_bigrams} bigrams:\n{bigrams}')

    numClustersList = [2, 3, 4, 5, 10, 20]
    col1 = 'normalized_tokens'
    col2 = 'normalized_tokens_str'
    n_words = 10

    for n_cluster in numClustersList:
        plotSilhouette(df, col1, col2, n_cluster, n_words)