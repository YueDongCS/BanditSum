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

## CNN/DailyMail Dataset
Instructions to download our preprocessed CNN/DailyMail Dataset can be found here.
https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

## Our Test Output:
https://drive.google.com/file/d/1tMiWuRzvDfHGwDILDXT2WFpyFcuHSK1n/view?usp=sharing

## Our Pre-trained Model:

Test data: https://drive.google.com/file/d/1PCl0VVfhlcEaz-eSc5alP_U8uaVQGc_P/view?usp=sharing

Pre-trained model: https://drive.google.com/file/d/13UB2GH_TT5SPQaYydnxYXYHClD4pbOIn/view?usp=sharing

### Installation
Our code requires PyTorch version >= 0.4.0. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

After PyTorch is installed, you do the followings 

1. Dowload the [url_lists dataset](https://github.com/abisee/cnn-dailymail)
2. Download the [chunked dataset](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail)
3. Download the Glove 100d(glove.6B.zip) [vocab](https://nlp.stanford.edu/projects/glove/)
4. Rename the `glove.6B.100d.txt` to `vocab_100d.txt`
5. Run `pickle_glove.py` to parse and pickle the Glove vectors.
4. Create a data directory at the same level of the BanditSum repository and place the datasets as the following
```bash
.
├── BanditSum
│   ├── dataLoader.py
│   ├── evaluate.py
│   ├── experiments.py
│   ├── helper.py
│   ├── log
│   │   └── placeholder
│   ├── main.py
│   ├── model
│   │   └── placeholder
│   ├── model.py
│   ├── pickle_glove.py
│   ├── README.html
│   ├── README.md
│   ├── reinforce.py
│   └── rougefonc.py
└── data
        ├── CNN_DM_pickle_data
        │   ├── chunked
        │   │   ├── test_000.bin
        │   │   ├── ...
        │   │   ├── train_287.bin
        │   │   ├── val_000.bin
        │   │   ├── ...
        │   │   └── val_287.bin
        │   ├── vocab_100d.txt
        |   └── vocab
        └── url_lists
                ├── all_test.txt
                ├── all_train.txt
                ├── all_val.txt
                ├── cnn_wayback_test_urls.txt
                ├── cnn_wayback_training_urls.txt
                ├── cnn_wayback_validation_urls.txt
                ├── dailymail_wayback_test_urls.txt
                ├── dailymail_wayback_training_urls.txt
                ├── dailymail_wayback_validation_urls.txt
                └── readme
```

3. Run `dataLoader.main`
4. Run `main.py`

