# tilse

__tilse__ is a toolkit for <b>ti</b>me<b>l</b>ine <b>s</b>ummarization and <b>e</b>valuation. In it
implements functionality for predicting timelines given a corpus and for evaluating timeline summaries against a
gold standard. For evaluation, it makes use of a family of ROUGE variants as described in Martschat and Markert (2017).

## Installation

__tilse__ is available on PyPi. You can install it via

```
pip install tilse
```
Dependencies (automatically installed by pip) are [NumPy](http://www.numpy.org/),
[SciPy](https://www.scipy.org/), [pathlib2](https://pypi.python.org/pypi/pathlib2/) (for Python < 3.4),
[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), sklearn, numba, textacy and nltk.
__tilse__ ships with an adapted version of [pyrouge](https://github.com/andersjo/pyrouge) and contains
[HeidelTime](https://github.com/HeidelTime/heideltime).

__tilse__ is written for use on Linux with Python 3.4+ but it also runs under Python 2.7.

## Documentation

* <a href="EVALUATION.md">using __tilse__ as an evaluation toolkit</a>
* <a href="SUMMARIZATION.md">predicting timelines with __tilse__</a>

## References

Sebastian Martschat and Katja Markert (2017). **Improving ROUGE for Timeline Summarization.** In *Proceedings
of the 15th Conference of the European Chapter of the Association for Computational Linguistics, volume 2: Short Papers*,
Valencia, Spain, 3-7 April 2017, pages 285-290.
[PDF](https://aclweb.org/anthology/E/E17/E17-2046.pdf)

Sebastian Martschat and Katja Markert (2018). **A Temporally Sensitive Submodularity Framework for Timeline Summarization.**
To appear in *Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL)*, Brussels, Belgium, 31 October-1 November 2018.