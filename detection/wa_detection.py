import os
import pickle
import numpy as np
from utils import utils
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
from detection.feature_selection import cv_train, detect, get_arch, get_archmap, arch_train



def cal(arg_dict, metaParameters):

    base_path = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    modeldirs = os.listdir(base_path)
    model_filepaths = [os.path.join(base_path, modeldir, 'model.pt') for modeldir in modeldirs]
    cls = [utils.get_class(os.path.join(base_path, modeldir, 'config.json')) for modeldir in modeldirs]
    cfg_dict={'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 0.03}
    holdoutratio=0.1
    num_cv_trials = arg_dict['num_cv_trials']

    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'cv_results')
    os.makedirs(results_dir, exist_ok=True)


    arch_map = get_archmap(model_filepaths)
    arch_weight_mappings = {}
    ISO_arch_classifiers = {}
    arch_classifiers = {}
    arch_xstats = {}


    #TODO refactor this code
    for arch, arch_inds in arch_map.items():
        print('starting arch', arch)
        # import pdb; pdb.set_trace()

        arch_fns = np.array([model_filepaths[i] for i in arch_inds])
        arch_classes = np.array([cls[i] for i in arch_inds])

        ns = arch_classes.shape[0]
        inds = np.arange(ns)
        split_ind = round((1-holdoutratio)*ns)

        lr_scores = []
        truths = []
        numSample = num_cv_trials
        for i in range(numSample):
            np.random.shuffle(inds)
            trinds = inds[:split_ind]
            vinds = inds[split_ind:]

            tr_fns = arch_fns[trinds]
            tr_cls = arch_classes[trinds]
            v_fns = arch_fns[vinds]
            v_cls = arch_classes[vinds]

            weight_mapping, classifier, xstats = arch_train(tr_fns, tr_cls, cfg_dict)

            pv = [detect(fn, weight_mapping, classifier, xstats) for fn in v_fns]
            print(f" AUC: {roc_auc_score(v_cls, pv)}, CE: {log_loss(v_cls, pv)}")
            lr_scores.append(pv)
            truths.append(v_cls)


        ISOce_scores = []

        for _ in range(numSample):
            ns = len(lr_scores)
            ind = np.arange(ns)
            np.random.shuffle(ind)
            split = round(  (1-holdoutratio)*ns)

            ptr = np.concatenate(lr_scores[ind[:split]])
            ptst = np.concatenate(lr_scores[ind[split:]])
            ytr = np.concatenate(truths[ind[:split]])
            ytst = np.concatenate(truths[ind[split:]])

            ir_model = IsotonicRegression(out_of_bounds='clip')
            ir_model.fit(ptr, ytr)
            p2tst = ir_model.transform(ptst)
            p2tst = np.clip(p2tst, 0.01, 0.99)

            ISOce_scores.append(log_loss(ytst, p2tst))

        print('new ISO CE', np.mean(ISOce_scores))


        weight_mapping, classifier, xstats = arch_train(arch_fns, arch_classes, cfg_dict)
        arch_weight_mappings[arch] = weight_mapping
        arch_classifiers[arch] = classifier
        ISO_arch_classifiers[arch] = ir_model
        arch_xstats[arch] = xstats

    # import pdb;pdb.set_trace()
    dump([arch_weight_mappings, arch_classifiers, ISO_arch_classifiers, arch_xstats], os.path.join(arg_dict['learned_parameters_dirpath'], 'wa_lr.joblib'))
    # import pdb; pdb.set_trace()
    # return mappings, classifiers, xstats


def detector(model_filepath,lr_path, metaParameters, gift_basepath):
    modelsplit =metaParameters["modelsplit"]
        
    mappings, classifiers, xstats_dict = load(lr_path)
    
    #TODO inferr the model archtecture and select the right weight_mapping from the data. This is a holder code

    arch = get_arch(model_filepath)
    weight_mapping = mappings[arch]
    classifier = classifiers[arch]
    xstats = xstats_dict[arch]
    assert not weight_mapping == mappings,  "Please select an appropriate weight_mapping"
    assert not classifier == classifiers, "Please select an appropriate classifier"
    assert not xstats == xstats_dict, "Please select an appropriate xstats"

    pv = detect(model_filepath, weight_mapping, classifier, xstats)
        
    return pv


def det(arg_dict, train=False):

    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname = "wa_lr.joblib"
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])

    lr_path = os.path.join(learned_parameters_dirpath, detname) 
    lr_path = os.path.join(basepath, lr_path)

    if train:
        cal(arg_dict, metaParameters)
    return lambda model_filepath: detector(model_filepath, lr_path, metaParameters, basepath)