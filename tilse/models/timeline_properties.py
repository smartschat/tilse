class TimelineProperties:
    """
    Represents timeline properties.

    Attributes:
        daily_summary_length (int): Daily summary length.
        num_dates (int): Number of dates in the timeline.
        num_sentences (int): Number of sentences in the timeline.
        start (datetime): First date of the timeline.
        end (datetime): Last date of the timeline.
    """
    def __init__(self,
                 daily_summary_length,
                 num_dates,
                 num_sentences,
                 start,
                 end):
        """
        Initializes timeline properties.

        Params:
            daily_summary_length (int): Daily summary length.
            num_dates (int): Number of dates in the timeline.
            num_sentences (int): Number of sentences in the timeline.
            start (datetime): First date of the timeline.
            end (datetime): Last date of the timeline.
        """
        self.daily_summary_length = daily_summary_length
        self.num_dates = num_dates
        self.num_sentences = num_sentences
        self.start = start
        self.end = end

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.daily_summary_length == other.daily_summary_length
                    and self.num_dates == other.num_dates
                    and self.num_sentences == other.num_sentences
                    and self.start == other.start
                    and self.end == other.end)
        else:
            return False

    def __hash__(self):
        return hash((self.daily_summary_length,
                     self.num_dates,
                     self.num_sentences,
                     self.start,
                     self.end))
