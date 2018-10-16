import numpy


class Token:
    """ Represents a token.

    Attributes:
        content (str): The content of the token.
        lemma (str): The lemma of the token.
        pos (str): The part-of-speech tag of the token.
        ner_type (str): The named entity type of the token.
        vector (numpy.ndarray): A vector representation of the token.
        date (datetime.datetime): The date of the token.
        time_span (str): The time span of the token. Either 'd' (day),
            'm' (month) or 'y' (year).
    """
    def __init__(self, content, lemma, pos, ner_type, vector, date, time_span):
        """ Construct a token.

        Params:
            content (str): The content of the token.
            lemma (str): The lemma of the token.
            pos (str): The part-of-speech tag of the token.
            ner_type (str): The named entity type of the token.
            vector (numpy.ndarray): A vector representation of the token.
            date (datetime.datetime): The date of the token.
            time_span (str): The time span of the token. Either 'd' (day),
                'm' (month) or 'y' (year).
        """
        self.content = content
        self.lemma = lemma
        self.pos = pos
        self.ner_type = ner_type

        self.vector = numpy.copy(vector)
        self.date = date
        self.time_span = time_span

    def __eq__(self, other):
        """ Checks for equality.

        Two tokens are equal if they agree in all attributes.

        Params:
            other (Object): Any object.
        Returns:
            True if both objects are Tokens and agree in all attributes.
        """
        if isinstance(other, self.__class__):
            return (self.content == other.content
                    and self.lemma == other.lemma
                    and self.pos == other.pos
                    and self.ner_type == other.ner_type
                    and numpy.array_equal(self.vector, other.vector)
                    and self.date == other.date
                    and self.time_span == other.time_span)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.content, self.lemma, self.pos, self.ner_type, self.date, self.time_span))

    def __str__(self):
        return self.content
