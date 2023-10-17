# atc-portuguese-di

This is the repository for the research article `Across the Atlantic: Distinguishing Between European and Brazilian Portuguese Dialects`.

### Setup

Although the notebooks contain cells to install the required packages, we recommend using a virtual environment to run the code equiped with the following packages:

```
pandas
nltk
joblib
sklearn
pytorch
datasets
transformers
numpy
warnings
re
```

### Dataset

- Download the PT-PT_BR paired plain text version of the TEDTALKS 2020 dataset from [here](https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/pt-pt_br.txt.zip).
- Extract the files to your preferred location. Edit [`sample.py`](/scripts/sample.py) so the path to the text files matches your local path. 
- Run the script to get the training, dev and test splits of the dataset.
- For feature extraction, use the [`data.ipynb`](/nbs/data.ipynb) notebook, with the paths to the split you want to use. This also includes filtering steps.
    - For length-based filtering, we suggest using the following thresholds:
        - <= 10 chars for single-sentence examples.
        - <= 40 chars for 4-sentence examples.
        - <= 400 chars for full-transcript examples.

### Models

#### Classical Techniques

For the classical techniques (Naive-Bayes, Logistic Regression, Adaptive Naive-Bayes), please use the [`classifier.ipynb`](/nbs/classifier.ipynb) notebook. This notebook is set up to be configurable, so please edit the parameters to your liking.

Don't forget to set the paths to the data files you want to use.

#### Transformers Experiments

We have included the code for the initial experiments with transformers in the [`lms.ipynb`](/nbs/lms.ipynb) notebook. This notebook contains only a short experiment as proof of concept.

Don't forget to set the paths to the data files you want to use.

### Others

If you find this work useful, please cite:

```
@article{redacted-2023-across,
    title={Across the Atlantic: Distinguishing Between European and Brazilian Portuguese Dialects},
    author={Redacted for review},
    journal={Redacted for review},
    year={2023}
}