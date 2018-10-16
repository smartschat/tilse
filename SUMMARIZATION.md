# Predicting Timelines with tilse

__tilse__ implements various models for predicting timelines. In particular, it implements a simple supervised
regression baseline, the model by Chieu and Lee (2004), and various models employing optimization of submodular functions.

## Getting and Processing Data

__tilse__ needs data in a specific format. The command `get-and-preprocess-data` downloads, processes and converts
existing timeline summarization corpora. Possible options for `get-and-preprocess-data` are `timeline17` and
`crisis`. The data is stored in a folder with the respective name in the current working directory (processing the
data may take a few hours).

## Predicting Timelines

Timelines can be predicted using the command `predict-timelines`, which takes a JSON configuration file as input (see
below). Evaluation results are printed to standard output, and output (containing predicted timelines, reference
timelines and evaluation results) are stored in the current working directory. Currently, timelines can only predicted
for corpora that have been downloaded and processed via `get-and-preprocess-data`.

## Configuration Files

`predict-timelines` expects a JSON configuration file. The file contains parameters needed for all models, and
model-specific parameters. The `config` folder in this repository contains configuration files that were used
to obtain the results in the paper.

The following example is a configuration file that describes a model using
optimization of submodular functions.

```
{
  "name": "tlsconstraints_timeline17",
  "algorithm": "submodular",
  "corpus": "/home/mitarb/martschat/timeline17",
  "restrict_topics_to": null,
  "date_selection": null,
  "assess_length": "average_length_in_sentences",
  "sentence_representation": "ChieuSentenceRepresentation",
  "keyword_mapping": {
    "bpoil": ["bp", "oil", "spill"],
    "egypt": ["egypt", "egyptian"],
    "finan": ["financial", "economic", "crisis"],
    "h1n1": ["h1n1", "swine", "flu"],
    "haiti": ["haiti", "quake", "earthquake"],
    "iraq": ["iraq", "iraqi"],
    "libya": ["libya", "libyan"],
    "mj": ["michael", "jackson"],
    "syria": ["syria", "syrian"],
    "yemen": ["yemen"]
  },
  "rouge_computation": "original",
  "properties": {
    "constraint": "is_valid_individual_constraints",
    "semantic_cluster": "clusters_by_similarity",
    "date_cluster": "clusters_by_date",
    "coefficients": [1, 1, 0, 0]
  }
}
```

### Parameters Needed for All Models

Name | Type | Valid values | Description
---- | ---- | ------------ | -----------
name | str  | any | Name of the experiments. Output of the experiment will be stored in a file called `name.obj`
algorithm | str | 'random', 'regression', 'chieu', 'submodular' | Which algorithm to use
corpus | str | any path | Location of the corpus to process
restrict_corpus_to | list(str) | entries: topics of the corpus | If not `null`, limit processing to specified topics
date_selection | str | functions in `tilse.models.date_selection` | Which date selection algorithm to use before predicting timelines. If `null`, do date selection jointly with summary generation
assess_length | str | functions in `tilse.models.assess_length` | Which daily summary length assession algorithm to use. Cannot be `null`
sentence_representation | str | classes in `tilse.representations_sentence_representations` | Which daily summary length assession algorithm to use. Cannot be `null`
keyword_mapping | map(str, list(str)) | Maps from topic names to list of keywords | If not `null`, only consider sentences that contain at least one of the specified keywords
rouge_computation | str | `original`, `reimplementation` | Which ROUGE implementation to use `original` uses the original ROUGE toolkit, `reimplementation` uses a Python reimplementation. The reimplementation is faster, but leads to slightly different results

### Parameters Needed for Specific Models

If the algorithm is set to 'chieu' or 'submodular', further model-specific parameters need to be given via
the 'properties' key.

#### Parameters for Chieu

Name | Type | Valid values | Description
---- | ---- | ------------ | -----------
interest_measure | str  | functions in `tilse.models.chieu.interest_measures` | Interest measure to use

#### Parameters for Submodular

Name | Type | Valid values | Description
---- | ---- | ------------ | -----------
constraint | str  | functions in `tilse.models.submodular.constraints` | Constraint to respect during greedy algorithm
semantic_cluster | str  | functions in `tilse.models.submodular.semantic_cluster_functions` | Cluster computation function on which semantic diversity is based
date_cluster | str  | functions in `tilse.models.submodular.date_cluster_functions` | Cluster computation function on which temporal diversity is based
coefficients | list(double) | lists of length 4 | coefficients for coverage, semantic diversity, temporal diversity, date references
