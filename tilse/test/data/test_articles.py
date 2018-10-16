import datetime
import unittest
import spacy

from tilse.data import documents


class TestArticle(unittest.TestCase):
    def setUp(self):
        nlp = spacy.load("en")

        self.my_article = documents.Document.from_xml(
            "2012-10-02",
            """<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
At <TIMEX3 tid="t58" type="TIME" value="2009-05-29T13:28">1:28 pm</TIMEX3> on <TIMEX3 tid="t55" type="DATE" value="2009-05-29">29 May 2009</TIMEX3> , ghostofsichuan wrote : A friend , who as been in the U.S. went home to visit her parents in Xinjiang .
Her father did not let her leave the house or have visitors for <TIMEX3 tid="t59" type="DURATION" value="P5D">five days</TIMEX3> and told her this was his social responsiblity .
</TimeML>""",
            nlp
        )

        self.my_article2 = documents.Document.from_xml(
            "2012-10-02",
            """<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
<TIMEX3 tid="t58" type="DATE" value="2012-09">last month</TIMEX3> I did something .
</TimeML>""",
            nlp
        )

    def test_publication_date(self):
        date = datetime.datetime.strptime("2012-10-02", '%Y-%m-%d').date()
        self.assertEqual(date, self.my_article.publication_date)

    def test_tokens(self):
        tokens = "At 1:28 pm on 29 May 2009 , ghostofsichuan wrote : A friend , who as been in the U.S. went home to visit her parents in Xinjiang .\nHer father did not let her leave the house or have visitors for five days and told her this was his social responsiblity ."
        self.assertEqual(tokens.split(), [str(tok) for
                                          sent in self.my_article for
                                          tok in sent.tokens])

    def test_token_dates(self):
        date1 = datetime.datetime.strptime(
            "2009-05-29T13:28", '%Y-%m-%dT%H:%M'
        ).date()

        date2 = datetime.datetime.strptime(
            "2009-05-29", '%Y-%m-%d'
        ).date()
        time_values = [
            None,
            date1,
            date1,
            None,
            date2,
            date2,
            date2] + [None]*46

        self.assertEqual(time_values,
                         [tok.date
                          for sent in self.my_article
                          for tok in sent])

    def test_sent_dates(self):
        self.assertEqual(
            datetime.datetime.strptime("2009-05-29", '%Y-%m-%d').date(),
            self.my_article.sentences[0].date
        )
        self.assertEqual(
            datetime.datetime.strptime("2012-10-02", '%Y-%m-%d').date(),
            self.my_article.sentences[1].date
        )
        self.assertEqual(
            datetime.datetime.strptime("2012-09-01", '%Y-%m-%d').date(),
            self.my_article2.sentences[0].date
        )

if __name__ == '__main__':
    unittest.main()
