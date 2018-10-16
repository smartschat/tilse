import numpy


class Sentence:
    """ Represents a sentence.

    Attributes:
        tokens (list(Token): A list of the sentence's tokens.
        vector (numpy.ndarray): A vector representation of the sentence.
        date (datetime.datetime): The date of the sentence.
        time_span (str): The time span of the sentence. Either 'd' (day),
            'm' (month) or 'y' (year).
    """
    def __init__(self, tokens, default_date):
        """ Constructs a sentence from tokens and date information.

        Vector representations are averaged over all tokens. Date and time
         span is set to the first token that has date and time span information.
         If no such token exists, `self.date` is set to `default_date` and
         `self.time_span` is set to `'d'`.

        Params:
            tokens (list(Token)): The tokens of the sentence.
            default_date (datetime.datetime): The date the sentence date
                should be set to if the tokens do not contain any
                date information.
        """
        self.tokens = tuple(tokens)

        self.vector = sum([t.vector for t in tokens]) / len(tokens)

        self.date = default_date
        self.time_span = "d"

        for t in tokens:
            if t.date:
                self.date = t.date
                self.time_span = t.time_span
                break

    def __eq__(self, other):
        """ Checks for equality.

        Two sentences are equal if they agree in all attributes.

        Params:
            other (Object): Any object.
        Returns:
            True if both objects are Sentences and agree in all attributes.
        """
        if isinstance(other, self.__class__):
            return (self.tokens == other.tokens
                    and numpy.array_equal(self.vector, other.vector)
                    and self.date == other.date
                    and self.time_span == other.time_span)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.tokens, self.date, self.time_span))

    def __str__(self):
        return " ".join([str(t) for t in self.tokens]).strip()

    def __iter__(self):
        """ Iterates over tokens in the sentence.
        """
        return iter(self.tokens)
