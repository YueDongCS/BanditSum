# BanditSum
This repository contains the pre-processed data and code for our EMNLP 2018 paper "[BanditSum: Extractive Summarization as a Contextual Bandit](https://arxiv.org/abs/1809.09672)". Please contact me at yue.dong2@mail.mcgill.ca for any question.

Please cite this paper if you use our code or data.
```
@inproceedings{dong2018banditsum,
  title={BanditSum: Extractive Summarization as a Contextual Bandit},
  author={Dong, Yue and Shen, Yikang and Crawford, Eric and van Hoof, Herke and Cheung, Jackie Chi Kit},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3739--3748},
  year={2018}
}
```

## New Updates:
It was recently discovered that our model can achieve better performance than the one reported in the paper (trained with two epochs) if trained to four epochs on CNN/DailyMail:

BanditSum reported in the paper: ROUGE-1 41.5  ROUGE-2 18.7  ROUGE-L 37.6

BanditSum trained after 4 epochs: ROUGE-1 41.68  ROUGE-2 18.78  ROUGE-L 38.00

## CNN/DailyMail Dataset
Instructions to download our preprocessed CNN/DailyMail Dataset can be found here.
https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

## Our Test Output:
https://drive.google.com/file/d/1tMiWuRzvDfHGwDILDXT2WFpyFcuHSK1n/view?usp=sharing

## Our Pre-trained Model:
Test data: https://drive.google.com/file/d/1PCl0VVfhlcEaz-eSc5alP_U8uaVQGc_P/view?usp=sharing

Pre-trained model: https://drive.google.com/file/d/13UB2GH_TT5SPQaYydnxYXYHClD4pbOIn/view?usp=sharing

The vocab file: https://drive.google.com/file/d/1W0QQkz5VNCk-YAnpSRc0ONFgR5SPGDA8/view?usp=sharing

### Installation
Our code is written with python 2.7. Please see the modification from David Beauchemin https://github.com/davebulaval/BanditSum  if you intend to convert the code to python 3.7.

Our code requires PyTorch version >= 0.4.0. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

After PyTorch is installed, you can run our model through main.py. 

