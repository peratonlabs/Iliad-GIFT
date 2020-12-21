# Iliad-GIFT
This repository contains code developed by the Perspecta Labs/Project Iliad "GIFT" team for the [IARPA/TrojAI program](https://pages.nist.gov/trojai/docs/about.html). 
Code was developed by Todd Huster and Emmanuel Ekwedike. 

Contact: thuster@perspectalabs.com

## Setup

```
conda env create -f environment.yml
conda activate trojai
```
Other steps:
 - Put the trojai models in 'data/round3models'. E.g., 'data/round3models/id-00000000/model.pt' should be a valid path
 - Run "mkdir ./test/scratch -p"
 - Put one model in test/model.pt. Results below based on using id-00000000/model.pt.
 - Put some example images in test/example.  Results below based on using the images from id-00000000/clean_example_data.
 - Install singularity (only necessary to build and run singularity containers)

## Run calibration
Calibrate on the first 10 round 3 models:
```
python ens_build.py --ens_name ens14 --modellist calibration/modelsets/r3_tiny.txt --overwrite
```
Calibrate on all round 3 models:
```
python ens_build.py --ens_name ens14 --modellist calibration/modelsets/r3_all_trainset.txt --overwrite
```
Note: there are other modelsets to select from in calibration/modelsets. 
The attacks and detection scores for each model are saved, so once all models have been run through calibration, subsequent calibration runs (e.g. on different subsets of the models) should be very fast. 

## Run outside of singularity
```
python ./ens14_detector.py --model_filepath ./test/model.pt --result_filepath ./test/output.txt --scratch_dirpath ./test/scratch --examples_dirpath ./test/example --gift_basepath ./
```
Output based on calibrating on 10 models and running on id-00000000:
```
starting attack on  ./test/model.pt with attack a3361e2d2009b4ede04bd11e69317dfa
computing uap score on ./test/model.pt with detector 6bbb12912c665d518fa9993a7c15fac9 and attack a3361e2d2009b4ede04bd11e69317dfa
starting attack on  ./test/model.pt with attack 5b996285c490494d33b33d8353ca6ef3
computing uap score on ./test/model.pt with detector d665b361798e2b6822847cfb92f27382 and attack 5b996285c490494d33b33d8353ca6ef3
starting attack on  ./test/model.pt with attack 21d8921ac790eb84794a4adede70e127
computing uap score on ./test/model.pt with detector bb85651d0e54aa21a2a2284ff1651e46 and attack 21d8921ac790eb84794a4adede70e127
starting attack on  ./test/model.pt with attack 4eb83820197948df84d851553eccf770
computing uap score on ./test/model.pt with detector 938473867a4aec5c540061548b4cf152 and attack 4eb83820197948df84d851553eccf770
starting attack on  ./test/model.pt with attack 685e87ee6d9013ac2dbcf1de4c5f80fd
computing uap score on ./test/model.pt with detector 7214b9a4fc00d2b78bf36df0f6e602a3 and attack 685e87ee6d9013ac2dbcf1de4c5f80fd
starting attack on  ./test/model.pt with attack 5f2592b7c83f4135b2ca4494a137e5b1
computing uap score on ./test/model.pt with detector a40a4bd6928aa0dd35f781e4ec5c97d9 and attack 5f2592b7c83f4135b2ca4494a137e5b1
computing uap score on ./test/model.pt with detector f8277cca04d78763c89e6ffbd0ecda02 and attack 5f2592b7c83f4135b2ca4494a137e5b1
starting attack on  ./test/model.pt with attack aed4e40e2aad1d813a784821fd6623d8
computing uap score on ./test/model.pt with detector 582580551b3cc5398b7a4e76077c3ee2 and attack aed4e40e2aad1d813a784821fd6623d8
Final Trojan Probability: 0.975
```

## Build & run container image:
```
sudo singularity build --force ./test/gift.simg ./singularity/ens14_detector.def
singularity run --nv ./test/gift.simg --model_filepath ./test/model.pt --result_filepath ./test/output.txt --scratch_dirpath ./test/scratch --examples_dirpath ./test/example
```

