import os
import sys
import numpy as np
import trinv
from trinv import trinv_sc
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import utils.sc_utils as sc_utils
import utils.utils as utils
import pickle
import random
import detection.cls

SC_EXAMPLES_PATH = 'config/reference_data/sc_data'

def cal(arg_dict, sc_dets, sc_cls):
    x = []
    y = []

    detstr = str(sc_dets)

    print(len(sc_dets))
    print(sc_dets)

    model_dirpath = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    tokenizers_path = os.path.join(arg_dict['configure_models_dirpath'], 'tokenizers')

    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'sc_results')
    os.makedirs(results_dir, exist_ok=True)

    scModels = utils.getModelSubset(base_path=model_dirpath, task="sc")
    scModels = list(scModels)
    print("SC models", scModels)
    random.shuffle(scModels)
    # scModels = ['id-00000034']
    for model_id in scModels:
        print("Current model: ", model_id)
        model_result = None

        single_model_path = os.path.join(model_dirpath, model_id)
        model_filepath = os.path.join(single_model_path, 'model.pt')
        configFile = utils.read_json(os.path.join(single_model_path,"config.json"))
        model_arch = configFile["model_architecture"]
        # print("arch: ", model_arch)
        tokenizer_filepath = os.path.join(tokenizers_path, utils.tok_dict_r9[model_arch])

        res_path = os.path.join(results_dir, model_id + '.p')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                saved_model_result = pickle.load(f)
            if str(saved_model_result["sc_dets"]) == detstr:
                model_result = saved_model_result

        if model_result is None:
            features = []
            for sc_det in sc_dets:
                det_loss, _, _ = trinv_sc.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=SC_EXAMPLES_PATH, tokenizer_filepath=tokenizer_filepath, scratch_dirpath=scratch_dirpath, **sc_det)
                features.append(det_loss)
            cls = utils.get_class(os.path.join(single_model_path, 'config.json'))
            model_result = {"sc_dets": sc_dets, 'cls': cls, 'features': features}
            with open(res_path, "wb") as f:
                pickle.dump(model_result, f)
        x.append(model_result['features'])
        y.append(model_result['cls'])
        print(f" feature: {model_result['features']} and class: {model_result['cls']}")
    print(f" X: {x} and y: {y}")
    xv = np.array(x)
    yv = np.array(y)

    cls_fun = getattr(detection.cls, sc_cls['name'])
    cls_model = cls_fun(sc_cls)

    cls_model.fit(xv, yv)


    #TODO put Regression paramter in the tunable parameter
    # lr_model = LogisticRegression(C=400.0, max_iter=10000, tol=1e-4)
    # lr_model.fit(xv, yv)

    y_prob =cls_model.predict_proba(xv)[:,1]
    auc = roc_auc_score(yv, y_prob)
    ce = log_loss(yv, y_prob)
    print(f"Training AUC: {auc} and CE: {ce}")

    detection.cls.sv_cls(sc_cls, cls_model, arg_dict['learned_parameters_dirpath'], 'sc_cls.p')
    # dump(lr_model, os.path.join(arg_dict['learned_parameters_dirpath'], 'sc_lr.joblib'))
    detection.cls.run_cv(arg_dict['num_cv_trials'], arg_dict['cv_test_prop'], cls_fun, sc_cls, xv, yv)



def detector(model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath, calpath, metaParameters, detpth=None, basepath='./'):
    feature = []
    for det_kwargs in metaParameters:

        det_loss, _, _ = trinv_sc.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=examples_dirpath, tokenizer_filepath=tokenizer_filepath, scratch_dirpath = scratch_dirpath, **det_kwargs)
        feature.append(det_loss)

    if calpath:
        calmodel = detection.cls.load_cls(calpath, args=None) # need to validate
        p = calmodel.predict_proba(np.array(feature).reshape(1,-1))[:,1]

    return p


def det(arg_dict, train = False):
    configure_models_dirpath = arg_dict["configure_models_dirpath"]
    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname = "sc_cls.p"
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])["sc_dets"]

    if train and configure_models_dirpath is None:
        sys.exit(f"Please provide configure_models_dirpath for training mode to work")

    detpth = None
    calpath = os.path.join(learned_parameters_dirpath, detname) 
    calpath = os.path.join(basepath, calpath)

    if train:
        assert False, 'cal is deprecated here'
        # cal(arg_dict, metaParameters)
    return lambda model_filepath, examples_dirpath, tokenizer_filepath, scratch_dirpath: detector(model_filepath=model_filepath, examples_dirpath=examples_dirpath,tokenizer_filepath=tokenizer_filepath, scratch_dirpath= scratch_dirpath, detpth=detpth, calpath=calpath, metaParameters=metaParameters, basepath=basepath)