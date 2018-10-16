def clusters_by_date(corpus):
    """
    Clusters sentences in the corpus by their dates. Sentences
    are in the same cluster if and only if they have the same 
    date.
    
    Params:
        corpus (tilse.data.corpora.Corpus): The corpus for which
            sentences should be clustered.
            
    Returns:
        list(int): Cluster indices for each sentence in the corpus
            (order of sentence is determined by iteration over
            documents and sentences).
    """
    
    dates_to_index = {}

    labels = []

    for doc in corpus:
        for sent in doc:
            date = sent.date
            if date not in dates_to_index:
                if not dates_to_index:
                    dates_to_index[date] = 0
                else:
                    dates_to_index[date] = max(dates_to_index.values()) + 1

            labels.append(dates_to_index[date])

    return labels
