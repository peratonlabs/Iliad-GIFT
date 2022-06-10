import os
import sys
import json
import torch



tok_dict_r9 = {
    "distilbert-base-cased": "distilbert-base-cased.pt",
    "google/electra-small-discriminator": "google-electra-small-discriminator.pt",
    "roberta-base": "roberta-base.pt",
    "deepset/roberta-base-squad2": "tokenizer-deepset-roberta-base-squad2.pt",
    }
#TODO
def getModelSubset(base_path='data/round9/models', task = "qa"):
    all_models = os.listdir(base_path)
    all_models.sort()
    subset_models = set()
    for model_id in all_models:
        model_dirpath = os.path.join(base_path,model_id)
        config_filepath = os.path.join(model_dirpath, 'config.json')
        with open(config_filepath, "r") as configfile:
            config = json.load(configfile)

        if "task_type" in config:
            if config["task_type"]== task:
                subset_models.add(model_id)
        else:
            subset_models.add(model_id)
            print("warning: no task type found. model will be included in subset")
    return subset_models


def read_json(truth_fn):
    with open(truth_fn) as f:
        jsonFile = json.load(f)
    return jsonFile


def read_truthfile(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    lc_truth = {k.lower(): v for k,v in truth.items()}
    return lc_truth


def get_class(truth_fn):
    truth = read_truthfile(truth_fn)
    return int(truth["poisoned"])


def getTask(model_filepath = None, configFile = None):
    """
    :param model_filepath: File path to the pytorch model file to be evaluated
    :param configFile: config file for the model
    """

    if not model_filepath and not configFile:
        sys.exit("Please set the value of either model_filepath or configFile. Exiting!")
    if configFile:
        configFile = read_json(configFile)
    else:
        pathToken = model_filepath.split("/")[:-1]
        pathToken.append("config.json")
        configFile  ="/".join(pathToken) 
        configFile = read_json(configFile)
    task = configFile["task_type"]
    return task


def getTaskV2(model_filepath = None):
    """
    :param model_filepath: File path to the pytorch model file to be evaluated
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_filepath, map_location=torch.device(device))
    task_type = model.__class__.__name__.split("For")[-1]
    if task_type == "QuestionAnswering":
        task = "qa"
    elif task_type == "SequenceClassification":
        task = "sc"
    elif task_type == "TokenClassification":
        task = "ner"
    return task