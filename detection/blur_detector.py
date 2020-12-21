
import sys
sys.path.append(".")
import utils.utils as utils
import numpy as np
import torch
from joblib import load, dump
import pickle

# EXAMPLES_FOLDER_NAME = 'example_data' # round2
# EXAMPLES_FOLDER_NAME = 'clean_example_data' # round3

# BASE_FOLDER = 'data/round2models' # round2
# BASE_FOLDER = 'data/round3models' # round3

def get_blur_mag(adv_path, filt_sz=15, sigma=2.0):
    """
    :param adv_path:
    :param filt_sz:
    :param sigma:
    :return:
    """
    filt = utils.gaussian_kernel_pt(filt_sz, sigma).cuda()
    data = np.load(adv_path)
    diffs = data[0] - data[1]
    data0 = torch.tensor(diffs).cuda()
    data1 = filt(data0).cpu().detach().numpy()
    mag_ranges = data1.max() - data1.min()
    mag_stds = data1.reshape(-1).std()
    # print(mag_ranges, mag_stds)
    return mag_stds


def detector(adv_path, ir_path):
    """
    :param adv_path:
    :param ir_path:
    :return: trojan probability
    """
    blur_mag = get_blur_mag(adv_path, sigma=2.0)
    ir_model = load(ir_path)
    trojan_probability = ir_model.transform([blur_mag])[0]
    print('Trojan Probability: {}'.format(trojan_probability))
    return trojan_probability


def cal(refn, out_fn, base_folder='data/round2models', example_folder_name='example_data'):
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
        try:
            with open(calpath, 'rb')as f:
                ldirs, pcal, mags = pickle.load(f)
            return ldirs, pcal, mags
        except:
            with open(calpath, 'rb')as f:
                ldirs, pcal = pickle.load(f)
            return ldirs, pcal

    mags = []
    y = []

    dirs = os.listdir(path=base_folder)
    for dir in dirs:
        adv_path = os.path.join(base_folder, dir, example_folder_name, refn)
        if os.path.exists(adv_path):
            mag = get_blur_mag(adv_path, sigma=2.0)
            truth_fn = os.path.join(base_folder, dir, 'config.json')
            cls = utils.get_class(truth_fn, classtype='binary', file=True)
            mags.append(mag)
            y.append(cls)


    ir_model = IsotonicRegression(out_of_bounds='clip')
    pcal = ir_model.fit_transform(mags, y)
    kld = log_loss(y, pcal)
    # print(kld)
    roc1 = roc_auc_score(y, np.array(pcal))
    print(out_fn, 'AUC:', roc1, 'KLD:', kld)

    # dump(ir_model, 'data/classifiers/blur' + '_ir.joblib')
    dump(ir_model, 'calibration/fitted/' + out_fn)
    pcal = pcal[np.argsort(dirs)]
    dirs.sort()
    with open(calpath,'wb') as f:
        pickle.dump([dirs, pcal, mags], f)

    return dirs, pcal, mags

