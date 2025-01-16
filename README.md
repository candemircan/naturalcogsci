# Read Me!

> Code for the project “Evaluating alignment between humans and neural network representations in image-based learning tasks”

The paper can be found [here](https://openreview.net/forum?id=8i6px5W1Rf&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions)).

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

### Abstract

> Humans represent scenes and objects in rich feature spaces, carrying information that allows us to generalise about category memberships and abstract functions with few examples. What determines whether a neural network model generalises like a human? We tested how well the representations of $86$ pretrained neural network models mapped to human learning trajectories across two tasks where humans had to learn continuous relationships and categories of natural images. In these tasks, both human participants and neural networks successfully identified the relevant stimulus features within a few trials, demonstrating effective generalisation. We found that while training dataset size was a core determinant of alignment with human choices, contrastive training with multi-modal data (text and imagery) was a common feature of currently publicly available models that predicted human generalisation. Intrinsic dimensionality of representations had different effects on alignment for different model types. Lastly, we tested three sets of human-aligned representations and found no consistent improvements in predictive accuracy compared to the baselines. In conclusion, pretrained neural networks can serve to extract representations for cognitive models, as they appear to capture some fundamental aspects of cognition that are transferable across tasks. Both our paradigms and modelling approach offer a novel way to quantify alignment between neural networks and humans and extend cognitive science into more naturalistic domains.


![task designs](./figures/task_designs.svg)

## Setup

You can install the Python & R dependencies as follows:

``` bash
git clone https://github.com/candemircan/naturalcogsci.git
cd naturalcogsci
pip install .
Rscript bin/install_packages.R
```

The code was tested on Python 3.9 and R 4.3

You will also need to install `jq`, which is used to parse JSON files in the bash scripts. It can be downloaded from [here](https://jqlang.github.io/jq/download/).

## Environment Variables

The code uses the environment variable `NATURALCOGSCI_ROOT` to determine
the root directory of the project. You can set this variable in your
`.bashrc` file (or whatever your shell rc file might be) as follows:

``` bash
export NATURALCOGSCI_ROOT=/path/to/naturalcogsci
```

For me, R could not read this variable from the shell, so I had to set
it in the `~/.Renviron` file as well as follows:

``` bash
NATURALCOGSCI_ROOT=/path/to/naturalcogsci
```

## Structure

``` bash
├── bin # bash, slurm, and python scripts for analyses as well as notebooks for visualisations
├── data
│   ├── ID # intrinsic dimensionality of representations
│   ├── THINGS # THINGS mental embeddings and stimulus IDs
│   ├── cka # centered kernel alignment with ground truth
│   ├── embedding_weights_and_binaries # weights for some external models
│   ├── features # extracted features from images for THINGS
│   ├── harmonization # harmonization alignment score of supervised models
│   ├── human_behavioural
│   │   ├── category_learning # human behavioural data for category learning task
│   │   └── reward_learning # human behavioural data for reward learning task
│   ├── learner_behavioural
│   │   ├── category_learning # model behavioural data for category learning task
│   │   └── reward_learning # model behavioural data for reward learning task
│   ├── nights # nights alignment score
│   ├── nights_features # extracted features from images for nights
│   ├── peterson # peterson alignment score
│   ├── peterson_features # extracted features from images for peterson
│   └── r2 # class separation of representations
├── experiments 
│   ├── category_learning # js, html, css code for category learning task
│   └── reward_learning # js, html, css code for reward learning task
├── figures 
├── naturalcogsci # python functions
└── stimuli # THINGS database
```

## Experiments

Both experiments are shared under the `experiments` folder. See the
`README.md` files in the respective folders for more information.

## Stimuli

If you want to extract the features from the images , you need to
download the THINGS database under the `stimuli` folder.

This can be done from the following link:

<https://things-initiative.org/uploads/THINGS/images.zip>

## Data

All the data created during this project is stored on the OSF [here](https://osf.io/h3t52/). You just need to put it all under the `data` folder.

### External Data

I cannot share the content under the THINGS and the embedding_weights_and_binaries folders, but you can download them from the following links:

**THINGS**

- unique_id.txt -> https://osf.io/2y463 (rename it to unique_id.csv)
- spose_embedding_49d_sorted.txt -> https://osf.io/4pgk8

**embedding_weights_and_binaries**

- crawl-300d-2M-subword.bin -> https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip (unzip it)
- The weights for the SLIP (and the corresponding CLIP and SimCLR) models from Meta can be downloaded from the URLs here: https://github.com/facebookresearch/SLIP?tab=readme-ov-file#results-and-pre-trained-models

## Citation

If you use our work, please cite our
[paper](https://openreview.net/forum?id=8i6px5W1Rf&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions)) as such:

``` bibtex
@inproceedings{demircan2024evaluating,
  title={Evaluating alignment between humans and neural network representations in image-based learning tasks},
  author={Demircan, Can and Saanum, Tankred and Pettini, Leonardo and Binz, Marcel and Baczkowski, Blazej M and Doeller, Christian F and Garvert, Mona M and Schulz, Eric},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
