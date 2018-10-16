import codecs
import logging
import os
import xml

from tilse.data import documents


class Corpus:
    """
    Represents a corpus.

    Attributes:
        name (str): The name of the corpus
        docs (list(Document)): The documents in the corpus.

    """
    def __init__(self, name, docs):
        """
        Constructs a corpus from documents.

        Params:
            name (str): The name of the corpus
            docs (list(Document)): The documents in the corpus.

        """
        self.name = name
        self.docs = docs

    def __iter__(self):
        """
        Iterates over documents in the corpus.

        Returns:
             An iterator over documents in the corpus.
        """
        return iter(self.docs)

    @staticmethod
    def from_folder(folder, nlp):
        """
        Creates a corpus from *.timeml files in a folder.

        Params:
            folder (str): Path to a folder with *.timeml files
            nlp (spacy.lang.en.English): An English spacy text-processing
                pipeline object.

        Returns:
            A corpus based on documents in `folder`.
        """
        created_documents = []

        for date in sorted(os.listdir(folder)):
            if os.path.isdir("/".join([folder, date])):
                for filename in sorted(os.listdir("/".join([folder, date]))):
                    if not filename.endswith(".timeml"):
                        continue

                    logging.debug("Now " + date + ", " + filename)

                    try:
                        my_doc = documents.Document.from_xml(
                            date,
                            codecs.open("/".join([folder, date, filename]),
                                        "r", "utf-8").read(),
                            nlp
                        )

                        if len(my_doc.sentences) == 0:
                            continue
                        else:
                            created_documents.append(
                                my_doc
                            )
                    except xml.etree.ElementTree.ParseError as e:
                        print(filename, e)
                        continue

        return Corpus(folder, created_documents)

    def filter_by_keywords_contained(self, keywords):
        """
        Filters corpus based on keywords.

        In particular, creates a new corpus by retaining only documents
        which contain at least one token whose lemma appears in the
        list of keywords.

        Params:
            keywords (list(str)): A list of keywords.

        Returns:
            A corpus with documents filtered by keyword containment.
        """
        my_docs = []
        for doc in self.docs:
            doc_sents = []
            for sent in doc:
                for tok in sent:
                    if tok.lemma in keywords:
                        doc_sents.append(sent)
                        break
            if len(doc_sents) > 0:
                new_doc = documents.Document(doc.publication_date, doc_sents)
                my_docs.append(new_doc)

        return Corpus(self.name + "_filtered_by_keywords", my_docs)

    def __eq__(self, other):
        """
        Checks for equality of name and documents.

        Params:
            other (Corpus): Another corpus.

        Returns:
            True if name and documents of corpora match, False otherwise.
        """
        if isinstance(other, self.__class__):
            return (self.name == other.name
                    and self.docs == other.docs)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.name, self.docs))
