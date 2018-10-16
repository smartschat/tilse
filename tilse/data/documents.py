import datetime
import logging
from xml.etree import ElementTree

import spacy

from tilse.data.sentences import Sentence
from tilse.data.tokens import Token


class Document:
    """ Represents a document.

    Attributes:
        publication_date (datetime.datetime): The publication date of the
            document.
        sentences (list(Sentence)): The sentences of the document.

    """
    def __init__(self, publication_date, sentences):
        """ Constructs a document from sentences and date information.

        Params:
            publication_date (datetime.datetime): The publication date of the
                document.
            sentences (list(Sentence)): The sentences of the document.
        """
        self.publication_date = publication_date
        self.sentences = tuple(sentences)

    @staticmethod
    def from_xml(publication_date, text, nlp):
        """ Constructs a document from a text in XML format.

        In particular, the text should conform to the the TimeML format
        (http://www.timeml.org/publications/timeMLdocs/timeml_1.2.1.html).

        Params:
            publication_date (str): The publication date of the
                document in format YYYY-MM-DD (e.g. 2011-03-12).
            text (str): The content of the document in TimeML format.
            nlp (spacy.lang.en.English): An English spacy text-processing
                pipeline object.

        Returns:
            A `Document` object which has publication date `publication_date`
            and sentences extracted from `text`.
        """
        logger = logging.getLogger(__name__)

        publication_date = datetime.datetime.strptime(
            publication_date, '%Y-%m-%d').date()

        tokens = []
        time_values = []
        time_spans = []

        root = ElementTree.fromstring(text)

        tokens.extend(root.text.split())
        time_values.extend([None] * len(tokens))
        time_spans.extend([None] * len(tokens))

        for time_tag in root:
            if time_tag.text is None:
                continue
            splitted_text = time_tag.text.split()
            tokens.extend(splitted_text)

            time_span = "d"

            if time_tag.attrib["type"] == "DATE":
                try:
                    value = [
                        datetime.datetime.strptime(
                            time_tag.attrib["value"], '%Y-%m-%d'
                        ).date()
                    ]
                except ValueError:
                    try:
                        value = [
                            datetime.datetime.strptime(
                                time_tag.attrib["value"], '%Y-%m'
                            ).date()
                        ]
                        time_span = "m"
                    except ValueError:
                        try:
                            value = [
                                datetime.datetime.strptime(
                                    time_tag.attrib["value"], '%Y'
                                ).date()
                            ]
                            time_span = "y"
                        except ValueError:
                            logger.warning("Could not parse date " +
                                           time_tag.attrib["value"])
                            value = [None]

            elif time_tag.attrib["type"] == "TIME":
                try:
                    value = [
                        datetime.datetime.strptime(
                            time_tag.attrib["value"], '%Y-%m-%dT%H:%M'
                        ).date()
                    ]
                except ValueError:
                    try:
                        value = [
                            datetime.datetime.strptime(
                                time_tag.attrib["value"], '%Y-%m-%dTMO'
                            ).date()
                        ]
                    except ValueError:
                        try:
                            value = [
                                datetime.datetime.strptime(
                                    time_tag.attrib["value"], '%Y-%m-%dTEV'
                                ).date()
                            ]
                        except ValueError:
                            try:
                                value = [
                                    datetime.datetime.strptime(
                                        time_tag.attrib["value"], '%Y-%m-%dTNI'
                                    ).date()
                                ]
                            except ValueError:
                                try:
                                    value = [
                                        datetime.datetime.strptime(
                                            time_tag.attrib["value"], '%Y-%m-%dTAF'
                                        ).date()
                                    ]
                                except ValueError:
                                    logger.warning("Could not parse date " +
                                                   time_tag.attrib["value"])
                                    value = [None]
            else:
                value = [None]

            time_values.extend(value * len(splitted_text))
            time_spans.extend(time_span * len(splitted_text))
            splitted_tail = time_tag.tail.split()
            tokens.extend(splitted_tail)
            time_values.extend([None] * len(splitted_tail))
            time_spans.extend([None] * len(splitted_tail))

        tokens = Document._process_tokens(tokens)

        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)

        nlp.tagger(doc)
        nlp.entity(doc)
        nlp.parser(doc)

        token_objects = []

        for token in doc:
            token_objects.append(
                Token(token.orth_,
                      token.lemma_,
                      token.tag_,
                      token.ent_type_,
                      token.vector,
                      time_values[token.i],
                      time_spans[token.i])
            )

        sentence_objects = []

        for sent in doc.sents:
            sent_tokens = token_objects[sent.start:sent.end]
            sentence_objects.append(Sentence(sent_tokens, publication_date))

        return Document(publication_date, sentence_objects)

    @staticmethod
    def _process_tokens(tokens):
        processed_tokens = []
        for tok in tokens:
            if tok == "-LRB-":
                processed_tokens.append("(")
            elif tok == "-RRB-":
                processed_tokens.append(")")
            elif tok == "``":
                processed_tokens.append('"')
            elif tok == "''":
                processed_tokens.append('"')
            elif tok == "`":
                processed_tokens.append("'")
            else:
                processed_tokens.append(tok)

        return processed_tokens

    def __str__(self):
        return "\n".join([str(s) for s in self.sentences]).strip()

    def __iter__(self):
        return iter(self.sentences)

    def __eq__(self, other):
        """
        Checks for equality of publication date and sentences.

        Params:
            other (Document): Another document.

        Returns:
            True if publication date and sentences of documents match,
            False otherwise.
        """
        if isinstance(other, self.__class__):
            return (self.publication_date == other.publication_date
                    and self.sentences == other.sentences)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.publication_date, self.sentences))
