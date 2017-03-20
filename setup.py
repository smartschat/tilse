from setuptools import setup


setup(
    name='tilse',
    version='0.1',
    packages=['tilse',
              'tilse.data',
              'tilse.evaluation',
              'tilse.test',
              'pyrouge',
              'pyrouge.tests'],

    url='http://github.com/smartschat/tilse',
    license='MIT',
    author='Sebastian Martschat',
    author_email='sebastian.martschat@gmail.com',
    description='A toolkit for timeline summarization and evaluation.',
    keywords=['NLP', 'CL', 'natural language processing',
                'computational linguistics', 'summarization',
                'text analytics'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        ],
    install_requires=['numpy', 'scipy', 'pathlib2', 'beautifulsoup4'],
    package_data={
        'pyrouge': ['tools/ROUGE-1.5.5/XML/DOM/*',
                    'tools/ROUGE-1.5.5/XML/Handler/*',
                    'tools/ROUGE-1.5.5/XML/*',
                    'tools/ROUGE-1.5.5/data/WordNet-1.6-Exceptions/*',
                    'tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions/*',
                    'tools/ROUGE-1.5.5/data/*',
                    'tools/ROUGE-1.5.5/*']
    },
    scripts=['bin/evaluate-timelines', 'bin/run-timeline-metrics-tests']
)
