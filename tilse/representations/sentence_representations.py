import math
from collections import defaultdict

import nltk
import numpy
from sklearn import metrics

from tilse.util.constants import STOPWORDS


class SentenceRepresentation:
    """
    Provides functionality for computing (vector-based) 
    sentence similarities.
    
    This class is abstract. Actual Functionality must be implemented
    by subclasses.
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            similarities for.
    """
    def __init__(self, corpus):
        """
        Initializes computing sentence representations.
        
        Params:
            corpus (tilse.data.corpora.Corpus): The corpus to compute
                sentence representations for.
        """
        self.corpus = corpus

    @classmethod
    def get_instance(cls, corpus):
        """
        Returns an object for computing sentence representations.
        
        Params:
            corpus (tilse.data.corpora.Corpus): The corpus to compute
                sentence representations for.
        """
        return cls(corpus)

    def _get_sents_with_representations(self):
        sents, sents_vec = [], []

        for doc in self.corpus:
            for sent in doc:
                sents.append(sent)
                sents_vec.append(self._get_sentence_vector(sent))

        return sents, numpy.array(sents_vec)

    def compute_pairwise_similarities(self):
        """
        Computes pairwise similarity between all sentences in the corpus.
        
        Uses cosine similarity, and normalizes between 0 and 1.
        
        Returns:
            numpy.array: A matrix of pairwise sentence similarities.
        """
        _, sents_vec = self._get_sents_with_representations()

        sims_temp = 1 - metrics.pairwise.pairwise_distances(sents_vec, metric="cosine")

        sims_temp += abs(sims_temp.min())
        sims_temp /= sims_temp.max()

        return sims_temp

    def _get_sentence_vector(self, sent):
        raise NotImplementedError("Needs to be implemented by subclass.")


class SpacySentenceRepresentation(SentenceRepresentation):
    """
    Provides functionality for computing (vector-based) 
    sentence similarities using vectors initially provided by spaCy.
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            sentence representations for.
    """
    def _get_sentence_vector(self, sent):
        return sent.vector


class SpacySentenceRepresentationIgnoringStopwords(SentenceRepresentation):
    """
    Provides functionality for computing (vector-based) 
    sentence similarities using vectors initially provided by spaCy.
    Ignores stopwords (cf. `tilse.util.constants`) when computing
    similarity.
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            sentence representations for.
    """
    def _get_sentence_vector(self, sent):
        vector = numpy.zeros(300)

        for tok in sent:
            if tok.lemma not in STOPWORDS:
                vector += tok.vector

        return vector


class ChieuSentenceRepresentation(SentenceRepresentation):
    """
    Provides functionality for computing (vector-based) 
    sentence similarites. Computes vectors using the
    "inverse date frequency" method from Chieu and Lee (2004):
    Query-based event extraction along a timeline".
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            sentence representations for.
    """
    def __init__(self, corpus):
        super(ChieuSentenceRepresentation, self).__init__(corpus)
        self.stemmer = nltk.stem.PorterStemmer()
        term_date_mapping = self._extract_term_date_mapping()
        self.mapping = self._convert_mapping_to_idf(term_date_mapping)

        self.stem_to_id = {}

        for i, stem in enumerate(sorted(self.mapping.keys())):
            self.stem_to_id[stem] = i

    def _extract_term_date_mapping(self):
        mapping = defaultdict(set)
        for doc in self.corpus:
            for sent in doc:
                for token in sent:
                    stem = token.lemma
                    try:
                        stem = self.stemmer.stem(token.lemma)
                    except IndexError:
                        pass

                    mapping[stem].add(sent.date)

        return mapping

    def _convert_mapping_to_idf(self, mapping):
        idf_mapping = {}

        all_dates = set()
        for doc in self.corpus:
            all_dates.add(doc.publication_date)
            for sent in doc:
                all_dates.add(sent.date)

        num_total_dates = len(all_dates)

        for term, dates in mapping.items():
            num_term_dates = len(dates)
            if num_term_dates < 3:
                num_term_dates = 3

            idf_mapping[term] = math.log(
                num_total_dates / num_term_dates
            )

        return idf_mapping

    def _get_sentence_vector(self, sent):
        tokens = []
        for tok in sent:
            # tried date check, doesn't improve performance
            if tok.lemma in STOPWORDS:
                continue
            else:
                tokens.append(tok)

        vec = numpy.zeros(len(self.stem_to_id))

        for tok in tokens:
            stem = tok.lemma

            try:
                stem = self.stemmer.stem(tok.lemma)
            except IndexError:
                pass

            stem_id = self.stem_to_id[stem]
            vec[stem_id] += self.mapping[stem]

        return vec


class DateWeightedChieuSentenceRepresentation(ChieuSentenceRepresentation):
    """
    Provides functionality for computing (vector-based) 
    sentence representations. Computes vectors using the
    "inverse date frequency" method from Chieu and Lee (2004):
    Query-based event extraction along a timeline". Reweights similarities
    linearly according to distance of the sentences' dates.
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            sentence representations for.
    """
    def compute_pairwise_similarities(self):
        """
        Computes pairwise similarity between all sentences in the corpus.
        
        Uses cosine similarity. Reweights similarities linearly according 
        to distance of the sentences' dates.
        
        Returns:
            numpy.array: A matrix of pairwise sentence similarities.
        """    
        sents, sents_vec = self._get_sents_with_representations()

        sims = 1 - metrics.pairwise.pairwise_distances(sents_vec, metric="cosine")

        for i, sent_a in enumerate(sents):
            for j, sent_b in enumerate(sents):
                sims[i][j] *= 1 / math.sqrt((math.fabs(sent_a.date.toordinal() - sent_b.date.toordinal()) + 1))

        return sims


class CutOffChieuSentenceRepresentation(ChieuSentenceRepresentation):
    """
    Provides functionality for computing (vector-based) 
    sentence representations. Computes vectors using the
    "inverse date frequency" method from Chieu and Lee (2004):
    Query-based event extraction along a timeline". Sets similarity
    of sentences with date distance > 10 to 0.
    
    Attributes: 
        corpus (tilse.data.corpora.Corpus): The corpus to compute
            sentence representations for.
    """
    def compute_pairwise_similarities(self):
        """
        Computes pairwise similarity between all sentences in the corpus.
        
        Uses cosine similarity. Sets similarity of sentences with date 
        distance > 10 to 0.
        
        Returns:
            numpy.array: A matrix of pairwise sentence similarities.
        """        
        sents, sents_vec = self._get_sents_with_representations()

        sims = 1 - metrics.pairwise.pairwise_distances(sents_vec, metric="cosine")

        for i, sent_a in enumerate(sents):
            for j, sent_b in enumerate(sents):
                if math.fabs(sent_a.date.toordinal() - sent_b.date.toordinal()) > 10:
                    sims[i][j] = 0

        return sims
