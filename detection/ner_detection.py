import os
import sys
import json
import numpy as np
import trinv
from trinv import trinv_ner
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import utils.ner_utils as ner_utils
import utils.utils as utils
import pickle
import random
import warnings
import detection.cls

warnings.filterwarnings("ignore")

NER_EXAMPLES_PATH = 'config/reference_data/ner_data'

def cal(arg_dict, ner_dets, ner_cls):
    x = []
    y = []

    detstr = str(ner_dets)

    print(len(ner_dets))
    print(ner_dets)

    model_dirpath = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    tokenizers_path = os.path.join(arg_dict['configure_models_dirpath'], 'tokenizers')

    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'ner_results')
    os.makedirs(results_dir, exist_ok=True)

    nerModels = utils.getModelSubset(base_path=model_dirpath, task="ner")
    nerModels = list(nerModels)
    random.shuffle(nerModels)
    print("NER models: ", nerModels)

    for model_id in nerModels:
        print("Current model: ", model_id)
        model_result = None

        single_model_path = os.path.join(model_dirpath, model_id)
        model_filepath = os.path.join(single_model_path, 'model.pt')
        configFile = utils.read_json(os.path.join(single_model_path,"config.json"))
        model_arch = configFile["model_architecture"]
        tokenizer_filepath = os.path.join(tokenizers_path, utils.tok_dict_r9[model_arch])

        res_path = os.path.join(results_dir, model_id + '.p')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                saved_model_result = pickle.load(f)
            if str(saved_model_result["ner_dets"]) == detstr:
                model_result = saved_model_result

        if model_result is None:
            features = []
            for ner_det in ner_dets:
                det_loss, _, _ = trinv_ner.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=NER_EXAMPLES_PATH, tokenizer_filepath=tokenizer_filepath, scratch_dirpath=scratch_dirpath, **ner_det)
                features.append(det_loss)
            cls = utils.get_class(os.path.join(single_model_path, 'config.json'))
            model_result = {"ner_dets": ner_dets, 'cls': cls, 'features': features}
            with open(res_path, "wb") as f:
                pickle.dump(model_result, f)
        x.append(model_result['features'])
        y.append(model_result['cls'])
        print(f" feature: {model_result['features']} and class: {model_result['cls']}")
    print(f" X: {x} and y: {y}")
    xv = np.array(x)
    yv = np.array(y)


    cls_fun = getattr(detection.cls, ner_cls['name'])
    cls_model = cls_fun(ner_cls)

    cls_model.fit(xv, yv)

    detection.cls.sv_cls(ner_cls, cls_model, arg_dict['learned_parameters_dirpath'], 'ner_cls.p')

    y_prob = cls_model.predict_proba(xv)[:,1]
    auc = roc_auc_score(yv, y_prob)
    ce = log_loss(yv, y_prob)
    print(f"Training AUC: {auc} and CE: {ce}")

    detection.cls.run_cv(arg_dict['num_cv_trials'], arg_dict['cv_test_prop'], cls_fun, ner_cls, xv, yv)







def detector(model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath, calpath, metaParameters, detpth=None, basepath='./'):

    feature = []
    for det_kwargs in metaParameters:

        det_loss, _, _ = trinv_ner.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=examples_dirpath, tokenizer_filepath=tokenizer_filepath, scratch_dirpath = scratch_dirpath, **det_kwargs)
        feature.append(det_loss)


    if calpath:
        calmodel = detection.cls.load_cls(calpath, args=None) # need to validate
        p = calmodel.predict_proba(np.array(feature).reshape(1,-1))[:,1]
    return p



def det(arg_dict, train=False):

    configure_models_dirpath = arg_dict["configure_models_dirpath"]
    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname = "ner_cls.p"

    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])["ner_dets"]

    if train and configure_models_dirpath is None:
        sys.exit(f"Please provide configure_models_dirpath for training mode to work")

    detpth = None
    calpath = os.path.join(learned_parameters_dirpath, detname)
    calpath = os.path.join(basepath, calpath)
   
   
    if train:
        assert False, 'cal is deprecated here'
        # cal(arg_dict, metaParameters)
    return lambda model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath: detector(model_filepath=model_filepath, examples_dirpath=examples_dirpath,tokenizer_filepath=tokenizer_filepath, scratch_dirpath= scratch_dirpath, detpth=detpth, calpath=calpath, metaParameters=metaParameters, basepath=basepath)




