# ArtQuest: Countering Hidden Language Biases in ArtVQA 
TODOs:
- [ ] Add the link to COVE repo: [https://cove.thecvf.com/datasets](https://cove.thecvf.com/datasets)

This repository contains the code used for the paper [ArtQuest: Countering Hidden Language Biases in ArtVQA](https://openaccess.thecvf.com/content/WACV2024/html/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.html) published at WACV 2024.

The following steps should be done in order to work.
Note: A modern GPU is needed for the trainings.

## Download of datasets
ArtQuest and SemArt need to be downloaded manually and extracted in the data directory.

The ArtQuest dataset is hosted on Zenodo: [https://zenodo.org/records/10453925](https://zenodo.org/records/10453925)

```
artquest
├── data
    └── artquest
    └── SemArt
```

## Install dependencies
Install the required packages and the directory as editable package.

`pip install -r requirements.txt`

`pip install -e .`

## Creation of SemArt cache
The cache is stores the image embeddings and contexts which are used.
To create it go to the clip directory (`cd clip`) and run `python3 make_semart_cache.py`.

## Training
The trainings of CLIP and T5 need to be done manually by going to their directories and running the training scripts from there.
Configuration and parameters are stored in the .json files and contain default parameters.

`cd clip
python3 train_clip.py -c clip-cfg.json`

`cd t5_training
python3 train_t5artquest.py -c t5artquest-cfg.json`

## Image-Context retrieval
To run the image context retrieval go the subdirectory of the retrieval module and run the script.
The config file is configs/semart_retrieval.yaml

`cd retrieval_module/image_text_retrieval`

`python3 image_text_retrieval.py`

## Evaluation
The evaluation needs all previous steps done. Go the the evaluation directory and run the evaluation script.

`cd evaluation`

`cd evaluate_artquest.py`

## Citation
If you use this work in academic publication, please cite the paper by Tibor Bleidt, Sedigheh Eslami, Gerard de Melo:

```
Bleidt, Tibor, Sedigheh Eslami, and Gerard de Melo. "ArtQuest: Countering Hidden Language Biases in ArtVQA." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024
```

BibTeX entry:

```
@inproceedings{bleidt2024artquest,
  title={ArtQuest: Countering Hidden Language Biases in ArtVQA},
  author={Bleidt, Tibor and Eslami, Sedigheh and de Melo, Gerard},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7326--7335},
  year={2024}
}
```
