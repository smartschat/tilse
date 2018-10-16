import unittest

import math
import datetime
from collections import defaultdict

import nltk
import spacy

import numpy

from tilse.data import documents, corpora
from tilse.models import chieu
from tilse.representations import sentence_representations

from scipy import sparse


class TestChieuSentRep(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en")

        first_article = documents.Document.from_xml(
            "2012-10-02",
            """<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
At <TIMEX3 tid="t58" type="TIME" value="2009-05-29T13:28">1:28 pm</TIMEX3> on <TIMEX3 tid="t55" type="DATE" value="2009-05-29">29 May 2009</TIMEX3> , ghostofsichuan wrote : A friend , who as been in the U.S. went home to visit her parents in Xinjiang .
Her father did not let her leave the house or have visitors for <TIMEX3 tid="t59" type="DURATION" value="P5D">five days</TIMEX3> and told her this was his social responsiblity .
</TimeML>""",
            self.nlp
        )

        second_article = documents.Document.from_xml(
            "2008-10-03",
            """<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
<TIMEX3 tid="t58" type="DATE" value="2008-10-02">Yesterday</TIMEX3> , I met you .
</TimeML>""",
            self.nlp
        )

        third_article = documents.Document.from_xml(
            "2008-10-05",
            """<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
.
</TimeML>""",
            self.nlp
        )

        self.corpus = corpora.Corpus("test", [first_article, second_article, third_article])
        self.chieu_sent_rep = sentence_representations.ChieuSentenceRepresentation(self.corpus)
        self.stemmer = nltk.stem.PorterStemmer()

    def test_extract_term_date_mapping(self):
        date1 = datetime.datetime.strptime("2009-05-29", '%Y-%m-%d').date()
        date2 = datetime.datetime.strptime("2012-10-02", '%Y-%m-%d').date()
        date3 = datetime.datetime.strptime("2008-10-02", '%Y-%m-%d').date()
        date4 = datetime.datetime.strptime("2008-10-05", '%Y-%m-%d').date()

        correct_mapping = defaultdict(set)

        correct_mapping.update({
            "at": {date1},
            "1:28": {date1},
            "pm": {date1},
            "on": {date1},
            "29": {date1},
            "may": {date1},
            "2009": {date1},
            ",": {date1, date3},
            "ghostofsichuan": {date1},
            "write": {date1},
            ":": {date1},
            "a": {date1},
            "friend": {date1},
            "who": {date1},
            "as": {date1},
            "be": {date1,date2},
            "in": {date1},
            "the": {date1, date2},
            "u.s.": {date1},
            "go": {date1},
            "home": {date1},
            "to": {date1},
            "visit": {date1},
            "-pron-": {date1, date2, date3},
            "parent": {date1},
            "xinjiang": {date1},
            ".": {date1, date2, date3, date4},
            "father": {date2},
            "do": {date2},
            "not": {date2},
            "let": {date2},
            "leave": {date2},
            "house": {date2},
            "or": {date2},
            "have": {date2},
            "visitor": {date2},
            "for": {date2},
            "five": {date2},
            "day": {date2},
            "and": {date2},
            "tell": {date2},
            "this": {date2},
            "social": {date2},
            "responsiblity": {date2},
            "yesterday": {date3},
            "meet": {date3},
        })

        correct_stemmed_mapping = defaultdict(set)

        for entry in correct_mapping:
            correct_stemmed_mapping[self.stemmer.stem(entry)].update(correct_mapping[entry])

        self.assertEqual(correct_stemmed_mapping, self.chieu_sent_rep._extract_term_date_mapping())

    def test_convert_mapping_to_idf(self):
        converted = self.chieu_sent_rep._convert_mapping_to_idf(
            self.chieu_sent_rep._extract_term_date_mapping()
        )

        self.assertEqual(math.log(5/4), converted["."])


if __name__ == '__main__':
    unittest.main()
