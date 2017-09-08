# tilse

__tilse__ is a toolkit for <b>ti</b>me<b>l</b>ine <b>s</b>ummarization and <b>e</b>valuation. In the current release it
implements functionality for evaluating timeline summaries against a gold standard. For doing so, it makes use of a
family of ROUGE variants as described in Martschat and Markert (2017).

## Installation

__tilse__ is available on PyPi. You can install it via

```
pip install tilse
```
Dependencies (automatically installed by pip) are [NumPy](http://www.numpy.org/) and
[SciPy](https://www.scipy.org/), [pathlib2](https://pypi.python.org/pypi/pathlib2/) (for Python < 3.4)
and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/). __tilse__ ships with an adapted
version of [pyrouge](https://github.com/andersjo/pyrouge).

__tilse__ is written for use on Linux with Python 3.4+ but it also runs under Python 2.7.

## Usage

__tilse__ can be used as a Python library as well as from the command line.

### __tilse__ as a library

```python
import datetime
import pprint

from tilse.data import timelines
from tilse.evaluation import rouge

evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1"])

predicted_timeline = timelines.Timeline({
    datetime.date(2010, 1, 1): ["Just a test .", "Another sentence ."],
    datetime.date(2010, 1, 3): ["Some more content .", "Even more !"]
})

groundtruth = timelines.GroundTruth(
    [
        timelines.Timeline({
            datetime.date(2010, 1, 2): ["Just a test ."],
            datetime.date(2010, 1, 4): ["This one does not match ."]
        }),
        timelines.Timeline({
            datetime.date(2010, 1, 5): ["Another timeline !"],
        })
    ]
)

pp = pprint.PrettyPrinter(indent=4)

print("contact")
pp.pprint(evaluator.evaluate_concat(predicted_timeline, groundtruth))
print("")
print("align, date-content costs")
pp.pprint(evaluator.evaluate_align_date_content_costs(predicted_timeline, groundtruth))
```

The above will have the output

```python
contact
{   'rouge_1': {   'f_score': 0.2222222222222222,
                   'precision': 0.16666666666666666,
                   'recall': 0.3333333333333333},
    'rouge_2': {'f_score': 0, 'precision': 0.0, 'recall': 0.0}}

align, date-content costs
{   'rouge_1': {   'f_score': 0.1111111111111111,
                   'precision': 0.08333333333333333,
                   'recall': 0.16666666666666666},
    'rouge_2': {'f_score': 0, 'precision': 0.0, 'recall': 0}}
```

### __tilse__ from the command line

You can run __tilse__ from the command line with the command `evaluate-timelines`. An example invocation is

```
evaluate-timelines -p timeline.txt -r reference_timeline1.txt reference_timeline2.txt -m agreement
```

where the timeline text files are in the format of the timelines shipped with the
[timeline17 dataset](http://l3s.de/~gtran/timeline/). Options for the parameter `-m` are `concat`, `agreement`,
`align_date_costs`, `align_date_content_costs` and `align_date_content_costs_many_to_one`.

## Running Tests

In order to run metric tests as described in our paper, you can use the command line tool `run-timeline-metrics-tests`.
The script processes sets of reference timelines of one or more topics. The tests can be invoked as follows:

```
run-timeline-metric-tests /path/to/timelines
```

where the timelines in `/path/to/timelines` are organized as in the following example:

```
/path/to/timelines/
    topic1/
        timeline1.txt
        timeline2.txt
    topic2/
        timeline1.txt
```

The numbers returned by the tests are slightly different from the numbers in the paper because the current
  version of __tilse__ implements a different variant of the `add` test and we run ROUGE with different parameters.

## References

Sebastian Martschat and Katja Markert (2017). **Improving ROUGE for Timeline Summarization.** In *Proceedings
of the 15th Conference of the European Chapter of the Association for Computational Linguistics, volume 2: Short Papers*
, Valencia, Spain, 3-7 April 2017, pages 285-290.  
[PDF](https://aclweb.org/anthology/E/E17/E17-2046.pdf)
