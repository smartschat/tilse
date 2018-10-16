from spacy.lang import en

STOPWORDS = en.STOP_WORDS.union({
    "?", "!", ".", ",", ":", "``", "''", "'s", "`", "--",
    ";", "(", ")", '"', "-LRB-", "-RRB-", "n't", "...",
    "-",
    "-PRON-"})
