import sys
sys.path.append(".")

import torch
import numpy as np
from joblib import load, dump
from utils import Filt_Model, readimages
import os
import pickle




def get_maxtarget(pairs):
    """
    This method looks at a set of (before_label, after_label) tuples.
     There are a few informative statistics from this set.
      - proportion of changed labels: P(before_label != after_label)
      - maximum proportional class gain: max_i(P(before_label != after_label AND after_label==i))
      - maximum class concentration of changed labels: max_i(P(after_label==i | before_label != after_label))
     Based on round 2 experiments, max proportional class gain is the best indicator, followed closely by
     proportion of changed labels.  Concentration was less reliable
    :param pairs: iterable of length 2 iterables.  (label before perturbation, label after perturbation)
    :return: maximum proportional class gain
    """

    # diffs = 0
    changes = {} # indexed by "after_label"
    tot = len(pairs)

    for c1, c2 in pairs:
        if c1 != c2:
            # diffs += 1  # number of changed labels
            if c2 not in changes:
                changes[c2] = 0
            changes[c2] += 1

    ch = [v for k, v in changes.items()]

    if len(ch) > 0:
        # ch1 = np.max(ch) / np.sum(ch)  # maximum class concentration of changed labels
        maxch = np.max(ch) / tot         # maximum proportional class gain
    else:
        # ch1 = 0
        maxch = 0

    # return 1-maxch # inverted to align polarity with other uap metric - nonfoolrate?
    return maxch



def get_foolrate_diff(model_path, example_dir, adv_path, pert_scale=1.0, nbatches=5, batchsz=100, use_confl=True, training=False):
    """
    computes the average foolrate of a set of perturbations in adv_path over different images
    foolrate is the percentage of samples that change class with and without the pertubration.
    this version of the method uses the difference between the two channels of adv_path
    in the future, it would probably make sense to just save the difference instead of both images
    :param model_path: (string) path to the model.pt file in question
    :param example_dir: (string) path to the example_data directory
    :param adv_path: (string) path to the numpy file containing the original and perturbed image pairs
    :param pert_scale: (float) desired scaling factor of the perturbation, i.e., x_pert = x + pert_scale*pert
    :param nbatches: (int) number of batches.  total samples used for computing foolrate is nbatches*batchsz
    :param batchsz: (int) batch size for evaluation, lowering this reduces memory consumption
    :param use_confl: (bool) use the slightly better "confluence" score provided by get_maxtarget
    :return: (float) foolrate or confluence score
    """


    data = readimages(example_dir)
    npoints = data.shape[0]

    model = torch.load(model_path).cuda()
    if training:
        model.train()
    else:
        model.eval()
    # model.train() ### REMOVE THIS!!!
    diffs = np.load(adv_path)
    diffs = diffs[1] - diffs[0]
    ndiffs = diffs.shape[0]

    inds = np.meshgrid([i for i in range(npoints)], [i for i in range(ndiffs)])
    inds = np.stack(inds, axis=2).reshape(-1, 2)
    np.random.shuffle(inds)

    nsamples = 0
    with torch.no_grad():
        diffs = torch.tensor(diffs).cuda()
        clean_data = torch.tensor(data).cuda()

        cls = []
        for i in range(int(np.ceil(clean_data.shape[0]/batchsz))):

            clean_batch = clean_data[i*batchsz:(i+1)*batchsz]

            cur_cls = model(clean_batch).argmax(axis=1).cpu().detach().numpy()
            cls.append(cur_cls)
        cls = np.concatenate(cls)

        foolrate = 0
        pairs = []

        for i in range(nbatches):
            curinds = inds[i*batchsz:(i+1)*batchsz]

            if curinds.shape[0]==0:
                break
            nsamples += curinds.shape[0]
            pert_data = clean_data[curinds[:,0]] + pert_scale*diffs[curinds[:,1]]
            pert_cls = model(pert_data).argmax(axis=1).cpu().detach().numpy()
            foolrate += (cls[curinds[:,0]]!=pert_cls).sum()
            for c1, c2 in zip(cls[curinds[:, 0]], pert_cls):
                pairs.append([c1, c2])

    if use_confl:
        return get_maxtarget(pairs)
    else:
        return foolrate/nsamples


def get_foolrate_filt(model_path, example_dir, filt_path, pert_scale=1.0, nbatches=5, batchsz=100, use_confl=True, training=False):
    """
    computes the average foolrate of a set of adversarial filters in filt_path over different images
    foolrate is the percentage of samples that change class with and without the pertubration.
    :param model_path: (string) path to the model.pt file in question
    :param example_dir: (string) path to the example_data directory
    :param filt_path: (string) path to the numpy file containing the filters
    :param pert_scale: (float) desired scaling factor of the filters, i.e., x_pert = x + conv(pert_scale*filter, x)
    :param nbatches: (int) number of batches.  total samples used for computing foolrate is nbatches*batchsz
    :param batchsz: (int) batch size for evaluation, lowering this reduces memory consumption
    :param use_confl: (bool) use the slightly better "confluence" score provided by get_maxtarget
    :return: (float) foolrate or confluence score
    """

    data = readimages(example_dir)
    npoints = data.shape[0]

    model = torch.load(model_path).cuda()
    if training:
        model.train()
    else:
        model.eval()
    filts = np.load(filt_path, allow_pickle=True)
    filts = np.concatenate([*filts])
    filts = filts.reshape(-1, 3, 4, filts.shape[-1], filts.shape[-1])
    filts *= pert_scale
    nfilts = filts.shape[0]

    filt_sz = filts.shape[-1]
    mod_model = Filt_Model(model, None, filt_sz=filt_sz, add_bias=True)

    inds = np.meshgrid([i for i in range(npoints)], [i for i in range(nfilts)])
    inds = np.stack(inds, axis=2).reshape(-1, 2)
    np.random.shuffle(inds)

    nsamples = 0
    with torch.no_grad():
        filts = torch.tensor(filts).cuda()
        clean_data = torch.tensor(data).cuda()

        cls = []
        for i in range(int(np.ceil(clean_data.shape[0]/batchsz))):

            clean_batch = clean_data[i*batchsz:(i+1)*batchsz]

            cur_cls = model(clean_batch).argmax(axis=1).cpu().detach().numpy()
            cls.append(cur_cls)
        cls = np.concatenate(cls)

        foolrate = 0
        pairs = []
        for i in range(nbatches):
            curinds = inds[i * batchsz:(i + 1) * batchsz]

            curinds_cuda = torch.tensor(curinds).cuda()
            # curinds = torch.tensor(inds[i * batchsz:(i + 1) * batchsz]).cuda()
            if curinds.shape[0]==0:
                break
            nsamples += curinds.shape[0]
            mod_model.set_images(clean_data[curinds_cuda[:, 0]])
            cur_filts = filts[curinds_cuda[:, 1]].reshape(-1, *filts.shape[2:])


            pert_cls = mod_model(cur_filts).argmax(axis=1).cpu().detach().numpy()
            foolrate += (cls[curinds[:,0]]!=pert_cls).sum()
            for c1, c2 in zip(cls[curinds[:, 0]], pert_cls):
                pairs.append([c1, c2])

    if use_confl:
        return get_maxtarget(pairs)
    else:
        return foolrate/nsamples


def detector(model_path, example_path, ir_path, adv_path=None, filt_path=None, pert_scale=2.0, use_confl=True, training=False, nbatches=5):
    """
    computes a probability of poisoning in the give model based on transferabilty
    :param model_path: (string) path to the model.pt file in question
    :param example_path: (string) path to the example_data directory
    :param ir_path: (string) path to the calibration model
    :param adv_path: (string) path to the numpy file containing the original and perturbed images
    :param filt_path: (string) path to the numpy file containing the filters (one of filt_path or adv_path must be set)
    :param pert_scale: (float) desired scaling factor of the filters/perturbation
    :param use_confl: (bool) use the slightly better "confluence" score provided by get_maxtarget
    :return: (float) foolrate or confluence score
    """

    # We may want to move to a kwargs system for handling all the pass through parameters.

    assert (adv_path is not None) != (filt_path is not None), "set either adv_path or filt_path"
    if adv_path is not None:
        foolrate = get_foolrate_diff(model_path, example_path, adv_path, pert_scale=pert_scale, use_confl=use_confl, training=training, nbatches=nbatches)
    else:
        foolrate = get_foolrate_filt(model_path, example_path, filt_path, pert_scale=pert_scale, use_confl=use_confl, training=training, nbatches=nbatches)
    ir_model = load(ir_path)
    trojan_probability = ir_model.transform([foolrate])[0]
    print('Trojan Probability: {}'.format(trojan_probability), 'precal: {}'.format(foolrate))
    return trojan_probability




# def detector(model_path, adv_path, pert_scale=2.0, ir_path='/round2uap_ir.joblib'):
#     foolrate = get_foolrate_diff(model_path, adv_path, pert_scale=pert_scale)
#     ir_model = load(ir_path)
#     trojan_probability = ir_model.transform([foolrate])[0]
#     print('Trojan Probability: {}'.format(trojan_probability))
#     return trojan_probability
#
#
# def detector_filt(model_path, example_path, filt_path, ir_path, pert_scale=10.0, use_confl=True):
#
#     # filt_path = adv_path[:-4]+filt_suffix
#     # nonfoolrate = get_nonfoolrate_filt(model_path, adv_path, filt_path, pert_scale=pert_scale)
#     foolrate = get_foolrate_filt(model_path, example_path, filt_path, pert_scale=pert_scale, use_confl=use_confl)
#     ir_model = load(ir_path)
#     trojan_probability = ir_model.transform([foolrate])[0]
#     print('Trojan Probability: {}'.format(trojan_probability))
#     return trojan_probability


def cal(out_fn, base_folder='data/round10models', example_folder_name='example_data', pert_scale=10., diff_fn=None, filt_fn=None, use_confl=False, training=False):
    """
    Calibrates a uap detector, saves it, and returns the calibrated probabilities
    :param out_fn: filename of output for calibration model
    :param base_folder: directory containing all the models (id-XXXXXXXX folders)
    :param pert_scale: (float) desired scaling factor of the filters/perturbation
    :param diff_fn: (string) filename of the numpy files containing the original and perturbed images
    :param filt_fn:  (string) filename of the numpy files containing the adversarial filters
    :param use_confl: (bool) use the slightly better "confluence" score provided by get_maxtarget
    :return:
    """

    assert (diff_fn is not None) != (filt_fn is not None), "set either diff_fn or filt_fn"

    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss, roc_auc_score
    import os
    from utils import utils

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
    # dirs=dirs[:100] # debugging
    for dir in dirs:
        example_path = os.path.join(base_folder, dir, example_folder_name)
        model_path = os.path.join(base_folder, dir, 'model.pt')

        if diff_fn is not None:
            adv_path = os.path.join(base_folder, dir, example_folder_name, diff_fn)
            mag = get_foolrate_diff(model_path, example_path, adv_path, pert_scale=pert_scale, use_confl=use_confl, training=training)
        else:
            filt_path = os.path.join(base_folder, dir, example_folder_name, filt_fn)
            mag = get_foolrate_filt(model_path, example_path, filt_path, pert_scale=pert_scale, use_confl=use_confl, training=training)

        truth_fn = os.path.join(base_folder, dir, 'config.json')
        cls = utils.get_class(truth_fn, classtype='binary', file=True)
        mags.append(mag)
        y.append(cls)

    mags = np.clip(mags, np.percentile(mags, 5), np.percentile(mags, 95))
    ir_model = IsotonicRegression(out_of_bounds='clip')
    pcal = ir_model.fit_transform(mags, y)
    kld = log_loss(y, pcal)
    roc1 = roc_auc_score(y, pcal)
    print(out_fn, 'AUC:', roc1, 'KLD:', kld)

    dump(ir_model, 'calibration/fitted/' + out_fn)
    pcal = pcal[np.argsort(dirs)]
    mags = np.array(mags)
    mags = mags[np.argsort(dirs)]
    dirs.sort()
    with open(calpath,'wb') as f:
        pickle.dump([dirs, pcal, mags], f)

    return dirs, pcal, mags



from collections import Counter


def get_targrate_filt(model_path, example_dir, filt_path, pert_scale=2.0, nbatches=5, batchsz=100, use_confl=False, percentile=0.9, training=False):


    data = readimages(example_dir)
    npoints = data.shape[0]

    model = torch.load(model_path).cuda()
    if training:
        model.train()
    else:
        model.eval()
    filts = np.load(filt_path, allow_pickle=True)
    filts = np.concatenate([*filts])
    filts = filts.reshape(-1, 3, 4, filts.shape[-1], filts.shape[-1])
    filts *= pert_scale
    nfilts = filts.shape[0]

    filt_sz = filts.shape[-1]
    mod_model = Filt_Model(model, None, filt_sz=filt_sz, add_bias=True)

    # mean_score = 0
    scores = []

    with torch.no_grad():

        filts = torch.tensor(filts).cuda()
        clean_data = torch.tensor(data).cuda()

        for j in range(nfilts):
            cur_filts = filts[j:j+1].repeat(batchsz,1,1,1,1).reshape(-1, *filts.shape[2:])
            pert_clses = []

            for i in range(int(np.ceil(npoints/batchsz))):
                cur_data = clean_data[i*batchsz:(i+1)*batchsz]
                mod_model.set_images(cur_data)

                pert_probs = mod_model(cur_filts[:cur_data.shape[0]*cur_data.shape[1]])
                pert_cls = pert_probs.argmax(axis=1).cpu().detach().numpy()
                pert_clses.append(pert_cls)

            ncls = pert_probs.shape[1]
            pert_clses = np.concatenate(pert_clses)

            c = Counter(pert_clses)

            score = max(list(c.values()))*ncls/pert_clses.shape[0]
            # print(score)
            # mean_score += score/nfilts
            scores.append(score)

    mean_score = np.mean(scores)

    perc = np.percentile(scores, percentile)

    return perc


def get_targrate_diff(model_path, example_dir, diff_path, pert_scale=2.0, nbatches=5, batchsz=100, use_confl=False, percentile=0.9, training=False):


    data = readimages(example_dir)
    npoints = data.shape[0]

    model = torch.load(model_path).cuda()
    if training:
        model.train()
    else:
        model.eval()
    diffs = np.load(diff_path)
    diffs = diffs[1] - diffs[0]
    ndiffs = diffs.shape[1]

    scores = []

    with torch.no_grad():
        diffs = torch.tensor(diffs).cuda()
        clean_data = torch.tensor(data).cuda()

        for j in range(ndiffs):

            curdiff = diffs[j:j+1]


            # cur_filts = filts[j:j+1].repeat(batchsz,1,1,1,1).reshape(-1, *filts.shape[2:])
            pert_clses = []

            for i in range(int(np.ceil(npoints/batchsz))):
                cur_data = clean_data[i*batchsz:(i+1)*batchsz]

                # pert_data = clean_data[curinds[:, 0]] + pert_scale * diffs[curinds[:, 1]]

                pert_probs = model(cur_data+pert_scale*curdiff)


                # mod_model.set_images(cur_data)
                #
                # pert_probs = mod_model(cur_filts[:cur_data.shape[0]*cur_data.shape[1]])
                pert_cls = pert_probs.argmax(axis=1).cpu().detach().numpy()
                pert_clses.append(pert_cls)

            ncls = pert_probs.shape[1]
            pert_clses = np.concatenate(pert_clses)

            c = Counter(pert_clses)

            score = max(list(c.values()))*ncls/pert_clses.shape[0]
            # print(score)
            # mean_score += score/nfilts
            scores.append(score)

    mean_score = np.mean(scores)

    perc = np.percentile(scores, percentile)

    return perc
