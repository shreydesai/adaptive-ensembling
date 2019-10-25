# Adaptive Ensembling: Unsupervised Domain Adaptation for Political Document Analysis

Code and datasets for our EMNLP 2019 paper "Adaptive Ensembling: Unsupervised Domain Adaptation for Political Document Analysis". If you found this project helpful, please consider citing our paper:

```bibtex
@inproceedings{desai2019unsupervised,
  author={Desai, Shrey and Sinno, Barea and Rosenfeld, Alex and Junyi Li, Jessy},
  title={Unsupervised Domain Adaptation for Political Document Analysis},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019},
}
```

## Datasets

Our [Corpus of Contemporary American English (COCA)](https://www.english-corpora.org/coca/) annotations are available in `datasets/coca_train.csv` and `datasets/coca_test.csv`, where they denote the train and test sets, respectively. Both files consist of the following structure:

```
docid,ag,pe,ir
152020,1,0,1
203568,0,0,0
164937,1,1,0
```

The `docid` field is the unique identifier given to each document in the COCA dataset. American Government (`ag`), Political Economy (`pe`), and International Relations (`ir`) are the category annotations we provide. Here, a `0` indicates the document *does not* have this label while a `1` indicates the document *does have* this label. Similarly, if a document does not have a `1` in any category, then it is labeled as non-political.

Next, we also provide a list of document identifiers that are labeled as "political" by our domain adaptation framework. This is available in `datasets/coca_politics.txt`.

For more information on our annotation process, label information, and domain adaptation methods, please see our paper for an in-depth discussion.
