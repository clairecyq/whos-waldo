# Model Card: Who's Waldo

We provide some accompanying information about the Who's Waldo model, inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).

## Model Details

The Who's Waldo model was developed by researchers at Cornell University for person-centric visual grounding. 

### Model Date

March 2021

### Model Version

There is one version of the model, found in this [repository](https://github.com/clairecyq/whos-waldo). 

### Model Type

The model uses a [UNITER](https://github.com/ChenRocks/UNITER) architecture as the backbone. The predictions and losses are computed using the similarity matrix output from UNITER. 

### Paper

[Who's Waldo Paper](https://arxiv.org/abs/2108.07253)

### Citation
```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Yuqing and Khandelwal, Apoorv and Artzi, Yoav and Snavely, Noah and Averbuch-Elor, Hadar},
    title     = {Who's Waldo? Linking People Across Text and Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1374-1384}
}
```
### License
[License](https://github.com/clairecyq/whos-waldo/blob/master/LICENSE)

### Where to send questions or comments about the model

Please use [this Google Form](https://forms.gle/EVHh8vVkGcAk6xeb6)


## Intended Use

### Primary Intended Uses and Users

The primary intended users of these models are AI researchers. We hope that this model will enable researchers to better understand and explore the new task of person-centric visual grounding, as well as improving on the task. 

### Out-of-Scope Use Cases

Any non-research use case of the model, whether commercial or not, is currently out of scope. 

## Metrics
Given a mapping produced by an algorithm for an input example, we evaluate by computing accuracy against ground truth links of referred people and detections.

## Training and Evaluation Data

### Dataset

The model was trained and evaluated on the [Who's Waldo Dataset](https://whoswaldo.github.io/dataset). 

### Preprocessing

The dataset preprocessing includes person detection and feature extraction from the images as well as coreference resolution from the text captions.

## Limitations
The model sometimes suffers from inability to comprehend complex interactions between people.
