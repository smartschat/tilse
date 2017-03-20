import datetime


class Timeline:
    """ Represent a timeline, which is a set of dates and corresponding summaries.

    Attributes:
        dates_to_summaries (dict(datetime.date, list(str))): A mapping of dates to summaries for the dates. Each
            summary is represented as a list of sentences. Each sentence is represented as a string where tokens
            are seperated by space.
    """
    def __init__(self, dates_to_summaries, filename=None):
        ''' Initialize a timelines.

        Args:
            dates_to_summaries (dict(datetime.date, list(str))): A mapping of dates to summaries for the dates. Each
            summary is represented as a list of sentences. Each sentence is represented as a string where tokens
            are seperated by space.
            filename (str): A filename associated with the timeline. Defaults to None.
        '''
        self.dates_to_summaries = {date: [s.strip() for s in summary] for date, summary in dates_to_summaries.items()}
        self.filename = filename

    def __str__(self):
        string_repr = ""
        for date in sorted(self.dates_to_summaries.keys()):
            string_repr += str(date) + "\n"
            for sent in self.dates_to_summaries[date]:
                string_repr += sent + "\n"
            string_repr += "--------------------------------\n"

        return string_repr


    def __eq__(self, other):
        return self.dates_to_summaries == other.dates_to_summaries

    def __lt__(self, other):
        return str(self) < str(other)

    def __iter__(self):
        return iter(self.dates_to_summaries.keys())

    def __getitem__(self, item):
        return self.dates_to_summaries.get(item, "")

    def __len__(self):
        return len(self.dates_to_summaries)

    def get_dates(self):
        return set(self.dates_to_summaries.keys())

    @staticmethod
    def from_file(my_file):
        """ Construct a timeline from a file.

        The file needs to be in the same format as the timelines from the timeline17 data set
        (http://l3s.de/~gtran/timeline/). Here is an excerpt:

        2010-09-19
        The ruptured well is finally sealed and `` effectively dead '' , says the top US federal official overseeing the disaster , Coast Guard Adm Thad Allen .
        --------------------------------
        2010-09-17
        BP pumps cement to seal the damaged well after it was intercepted by a relief well .
        --------------------------------

        Args:
            my_file (file): The file which contains the timeline.

        Returns:
            A timeline object that represents the timeline described in the file.
        """
        dates_to_summaries = {}

        date = None
        summary = ""

        for line in my_file.readlines():
            line = line.strip()
            if line == "--------------------------------":
                dates_to_summaries[date] = summary.strip().split("\n")
                date = None
                summary = ""
                continue
            else:
                try:
                    date = datetime.datetime.strptime(line, '%Y-%m-%d').date()
                except ValueError:
                    summary += line + "\n"

        return Timeline(dates_to_summaries, my_file.name)


class GroundTruth:
    """ Represent a collection of (reference) timelines.

    Thic class is designed to be used for evaluation.

    Attributes:
        timelines (collection(Timeline)): A collection of timelines.
    """
    def __init__(self, timelines):
        """ Initialize from a collection of reference timelines..

        Attributes:
            timelines (list(Timeline)): A collection of timelines.
        """
        self.timelines = timelines

    def get_dates(self):
        """ Return all dates which are in any ot the contained timelines.

        Returns:
            A set of datetime.date objects. This set contains all dates that are
            in any of the contained timelines.
        """
        all_keys = set()

        for tl in self.timelines:
            all_keys.update(tl.dates_to_summaries.keys())

        return all_keys

    def __iter__(self):
        all_keys = set()

        for tl in self.timelines:
            all_keys.update(tl.dates_to_summaries.keys())

        return all_keys

    def __str__(self):
        output = ""

        for i, tl in enumerate(self.timelines):
            output += str(i) + "\n\n" + str(tl) + "\n\n"

        return output

    def __getitem__(self, item):
        return {str(i): timeline[item] for i, timeline in enumerate(self.timelines)}