# Iliad-GIFT (r5)
This repository contains code developed by the Perspecta Labs/Project Iliad "GIFT" team for the [IARPA/TrojAI program](https://pages.nist.gov/trojai/docs/about.html). 
Code was developed by Todd Huster and Emmanuel Ekwedike. 

Contact: thuster@perspectalabs.com

## Setup

Install Anaconda, then run:

```
conda create --name trojai_rnd5 python=3.8 -y
conda activate trojai_rnd5

pip install trojai
conda install jsonpickle
```

Other steps:
 - Put the round 5 trojai models in 'data/round5models' (or soft link)
 - Put one model in test/model.pt
 - Install singularity.

## Run calibration

```
python r5cal.py
```

## Run outside of singularity
```
python ./jac_detector.py --model_filepath ./test/model.pt --result_filepath ./test/output.txt --scratch_dirpath ./test/scratch --gift_basepath ./ --cls_token_is_first --tokenizer_filepath /123fakepath --embedding_filepath /123fakepath
```

Example output:
```
val auc: 0.9222533573014698 pre-cal ce: 0.4241356229651577 post-cal ce: 0.32707997069706846 0.3598035352425011
```

## Build & run container image
```
sudo singularity build --force ./test/gift.simg ./singularity/jac_detector.def
singularity run --nv ./test/gift.simg --model_filepath ./test/model.pt --result_filepath ./test/output.txt --scratch_dirpath ./test/scratch --cls_token_is_first --tokenizer_filepath /home/trojai/tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath /home/trojai/embeddings/DistilBERT-distilbert-base-uncased.pt
```






