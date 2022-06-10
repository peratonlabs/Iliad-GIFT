# Iliad-GIFT (Round 9)
This repository contains code developed by the Peraton Labs' (formerly Perspecta Labs) Project Iliad "GIFT" team for the [IARPA/TrojAI program](https://pages.nist.gov/trojai/docs/about.html). 
Code was developed by Todd Huster and Emmanuel Ekwedike. 

Contact: thuster@peratonlabs.com

## Setup

Install Anaconda, then run:

```
conda env create -f conda_r9_env.yml
conda activate trojai9
```

## Calibration

```
python nlp_detector.py --metaparameters_filepath ./config/metaparameters_scexp.json --configure_mode --configure_models_dirpath /PATH/TO/ROUND9/DATA --task all
```

## Building Singularity container
Install Singularity, then run

```
sudo singularity build --force /PATH/TO/OUTPUT/gift_r9.simg ./singularity/nlp_detector.def
```







