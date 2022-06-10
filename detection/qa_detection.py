import os
import sys
import json
import numpy as np
import trinv
from trinv import trinv_qa
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import utils.qa_utils as qa_utils
import utils.utils as utils
import pickle
import random
import detection.cls

QA_EXAMPLES_PATH = 'config/reference_data/qa_data'

def cal(arg_dict, qa_dets, qa_cls):
    x = []
    y = []

    detstr = str(qa_dets)

    print(len(qa_dets))
    print(qa_dets)

    model_dirpath = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    tokenizers_path = os.path.join(arg_dict['configure_models_dirpath'], 'tokenizers')

    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'qa_results')
    os.makedirs(results_dir, exist_ok=True)

    qaModels = utils.getModelSubset(base_path=model_dirpath, task="qa")
    qaModels = list(qaModels)
    random.shuffle(qaModels)

    for model_id in qaModels:
        model_result = None

        single_model_path = os.path.join(model_dirpath, model_id)
        model_filepath = os.path.join(single_model_path, 'model.pt')
        configFile = utils.read_json(os.path.join(single_model_path,"config.json"))
        model_arch = configFile["model_architecture"]
        tokenizer_filepath = os.path.join(tokenizers_path, qa_utils.tok_dict_r9[model_arch])

        res_path = os.path.join(results_dir, model_id + '.p')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                saved_model_result = pickle.load(f)
            if str(saved_model_result["qa_dets"]) == detstr:
                model_result = saved_model_result

        if model_result is None:
            features = []
            for qa_det in qa_dets:
                det_loss, _, _ = trinv_qa.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=QA_EXAMPLES_PATH, tokenizer_filepath=tokenizer_filepath, scratch_dirpath=scratch_dirpath, **qa_det)
                features.append(det_loss)
            cls = utils.get_class(os.path.join(single_model_path, 'config.json'))
            model_result = {"qa_dets": qa_dets, 'cls': cls, 'features': features}
            with open(res_path, "wb") as f:
                pickle.dump(model_result, f)
        x.append(model_result['features'])
        y.append(model_result['cls'])
        print(f" feature: {model_result['features']} and class: {model_result['cls']}")
    print(f" X: {x} and y: {y}")
    xv = np.array(x)
    yv = np.array(y)

    #TODO put Regression paramter in the tunable parameter
    # lr_model = LogisticRegression(C=400.0, max_iter=10000, tol=1e-4)

    cls_fun = getattr(detection.cls, qa_cls['name'])
    cls_model = cls_fun(qa_cls)

    cls_model.fit(xv, yv)


    y_prob =cls_model.predict_proba(xv)[:,1]
    auc = roc_auc_score(yv, y_prob)
    ce = log_loss(yv, y_prob)
    print(f"Training AUC: {auc} and CE: {ce}")

    detection.cls.sv_cls(qa_cls, cls_model, arg_dict['learned_parameters_dirpath'], 'qa_cls.p')
    # dump(lr_model, os.path.join(arg_dict['learned_parameters_dirpath'], 'qa_lr.joblib'))
    detection.cls.run_cv(arg_dict['num_cv_trials'], arg_dict['cv_test_prop'], cls_fun, qa_cls, xv, yv)



def detector(model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath, calpath, metaParameters, detpth=None, basepath='./'):

    feature = []
    for det_kwargs in metaParameters:

        det_loss, _, _ = trinv_qa.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=examples_dirpath, tokenizer_filepath=tokenizer_filepath, scratch_dirpath = scratch_dirpath, **det_kwargs)
        feature.append(det_loss)


    if calpath:
        # calmodel = load(calpath)
        calmodel = detection.cls.load_cls(calpath, args=None)  # need to validate
        p = calmodel.predict_proba(np.array(feature).reshape(1,-1))[:,1]
    return p


def det(arg_dict, train=False):

    configure_models_dirpath = arg_dict["configure_models_dirpath"]
    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname = "qa_cls.p"
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])["qa_dets"]

    if train and configure_models_dirpath is None:
        sys.exit(f"Please provide configure_models_dirpath for training mode to work")

    detpth = None
    calpath = os.path.join(learned_parameters_dirpath, detname)
    calpath = os.path.join(basepath, calpath)
   
   
    if train:
        assert False, 'cal is deprecated here'
        # cal(arg_dict, metaParameters)
    return lambda model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath: detector(model_filepath=model_filepath, examples_dirpath=examples_dirpath,tokenizer_filepath=tokenizer_filepath, scratch_dirpath= scratch_dirpath, detpth=detpth, calpath=calpath, metaParameters=metaParameters, basepath=basepath)