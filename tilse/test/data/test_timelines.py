
import datetime
import os
import unittest

from tilse.data import timelines


class TestTimeline(unittest.TestCase):
    def test_from_file(self):
        self.maxDiff = None
        dates_to_summaries = {
            datetime.datetime.strptime("2010-09-19",
                                       '%Y-%m-%d').date(): ["The ruptured well is finally sealed and `` effectively dead '' , says the top US federal official overseeing the disaster , Coast Guard Adm Thad Allen ."],
            datetime.datetime.strptime("2010-09-17",
                                       '%Y-%m-%d').date(): ["BP pumps cement to seal the damaged well after it was intercepted by a relief well ."],
        }
        self.assertEqual(dates_to_summaries,
                         timelines.Timeline.from_file(open(os.path.dirname(os.path.abspath(__file__)) + "/resources/timeline.txt")).dates_to_summaries)

if __name__ == '__main__':
    unittest.main()