from sklearn import cluster


def clusters_by_similarity(corpus):
    """
    Clusters sentences by semantic similarity. Clustering
    is performed using kmeans with 0.2*number of sentences
    clusters, relying on sentence vector representations
    provided by spaCy.
    
    Params:
        corpus (tilse.data.corpora.Corpus): The corpus for which
            sentences should be clustered.
            
    Returns:
        list(int): Cluster indices for each sentence in the corpus
            (order of sentence is determined by iteration over
            documents and sentences).
    """
    all_sents_vectors = [
        sent.vector for doc in corpus for sent in doc
        ]

    num_clusters = int(0.2 * len(all_sents_vectors))

    kmeans = cluster.MiniBatchKMeans(num_clusters, random_state=23)
    kmeans.fit_predict(all_sents_vectors)

    return kmeans.labels_
