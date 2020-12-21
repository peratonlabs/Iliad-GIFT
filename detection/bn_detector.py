

import utils
from joblib import load, dump
import numpy as np
import pickle

def get_accdrop(model_path, examples_dirpath):
    batch_sz = 64

    acc_eval = utils.get_model_accuracy(model_path, examples_dirpath, batch_sz=batch_sz,
                                        nbatches=10, train_mode=False)
    acc_train = utils.get_model_accuracy(model_path, examples_dirpath, batch_sz=batch_sz,
                                         nbatches=10, train_mode=True)
    acc_drop = acc_eval - acc_train
    return acc_drop


def detector(model_path, examples_dirpath, ir_path):
    acc_drop = get_accdrop(model_path, examples_dirpath)

    ir_model = load(ir_path)
    trojan_probability = ir_model.transform([acc_drop])[0]

    if acc_drop>0.09 and acc_drop<0.15:
        trojan_probability=.98
    if trojan_probability > .995:
        trojan_probability = .995
    if trojan_probability < 0.3859:
        trojan_probability = 0.3859

    print('Trojan Probability: {}'.format(trojan_probability))
    return trojan_probability


def cal(out_fn, base_folder='data/round3models', example_folder_name='clean_example_data'):
    """
    :param refn:
    :param out_fn:
    :param base_folder:
    :return:
    """

    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss, roc_auc_score
    import os

    calpath = 'calibration/data/' + out_fn + '_caldata.p'

    if os.path.exists(calpath):
        with open(calpath, 'rb')as f:
            ldirs, pcal = pickle.load(f)
        return ldirs, pcal

    acc_drops = []
    y = []

    dirs = os.listdir(path=base_folder)
    for dir in dirs:
        example_path = os.path.join(base_folder, dir, example_folder_name)
        model_path = os.path.join(base_folder, dir, 'model.pt')
        acc_drop = get_accdrop(model_path, example_path)
        truth_fn = os.path.join(base_folder, dir, 'config.json')
        cls = utils.get_class(truth_fn, classtype='binary', file=True)
        acc_drops.append(acc_drop)
        y.append(cls)


    ir_model = IsotonicRegression(out_of_bounds='clip')
    pcal = ir_model.fit_transform(acc_drops, y)
    kld = log_loss(y, pcal)
    # print(kld)
    roc1 = roc_auc_score(y, np.array(pcal))
    print(out_fn, 'AUC:', roc1, 'KLD:', kld)

    # dump(ir_model, 'data/classifiers/blur' + '_ir.joblib')
    dump(ir_model, 'calibration/fitted/' + out_fn)
    pcal = pcal[np.argsort(dirs)]
    dirs.sort()
    with open(calpath,'wb') as f:
        pickle.dump([dirs, pcal], f)

    return dirs, pcal

