# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import random
import torch
import warnings
# import lipsch
from joblib import dump, load
import utils
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

archmap = {'GruLinear':0, 'Linear':1, 'LstmLinear':2}

warnings.filterwarnings("ignore")
TMPFN = 'calibration/fitted/jacdata.p'



def cal(modellist, holdoutratio=0.2, n_estimators=10000, tmppth=None, tmp_overwrite=False,detname=''):
    model_dirpaths = utils.get_modeldirs(modellist, usefile=True)

    x = []
    y = []

    if tmppth is not None and os.path.exists(tmppth):
        with open(tmppth,'rb') as f:
            xsv = pickle.load(f)
    else:
        xsv = {}
    for model_dirpath in model_dirpaths:
        if model_dirpath in xsv:
            feats = xsv[model_dirpath]
        else:
            print('getting feats from', model_dirpath)
            feats = get_jac_feats(os.path.join(model_dirpath, 'model.pt'), nsamples=10000)
            # feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
            # feats = np.concatenate(feats)
            xsv[model_dirpath] = feats

        # lcs1, lcs2, lcsi = feats
        # qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])

        cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))

        x.append(feats)
        y.append(cls)

    if tmppth is not None and (tmp_overwrite or not os.path.exists(tmppth)):
        with open(tmppth,'wb') as f:
            pickle.dump(xsv, f)
    x = np.stack(x, 0)
    y = np.array(y)

    ind = np.arange(len(y))
    np.random.shuffle(ind)

    split = round(len(y) * (1-holdoutratio))

    xtr = x[ind[:split]]
    xv = x[ind[split:]]
    ytr = y[ind[:split]]
    yv = y[ind[split:]]

    model = RandomForestClassifier(n_estimators=n_estimators)
    # model = CalibratedClassifierCV(model)

    model.fit(xtr, ytr)

    pv = model.predict_proba(xv)

    pv = pv[:, 1]

    vroc = roc_auc_score(yv, pv)
    vkld = log_loss(yv, pv)

    # lr_model = LogisticRegression(C=100)
    lr_model = LogisticRegression(C=100, max_iter=10000, tol=1e-4)
    lr_model.fit(pv.reshape(-1,1), yv)
    pv3 = lr_model.predict_proba(pv.reshape(-1,1))[:, 1]


    vkld3 = log_loss(yv, pv3)


    ir_model = IsotonicRegression(out_of_bounds='clip')
    pv2 = ir_model.fit_transform(pv, yv)

    vkld2 = log_loss(yv, pv2)
    #
    print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal ce:',vkld2, vkld3)



    # dump(model, detname)
    dump(model, 'calibration/fitted/' + detname + '_rf.joblib')
    dump(ir_model, 'calibration/fitted/' + detname + '_ir.joblib')
    dump(lr_model, 'calibration/fitted/' + detname + '_lr.joblib')


def evaldet(modellist, detpth, calpath=None):
    model_dirpaths = utils.get_modeldirs(modellist, usefile=True)

    probs = []
    ys = []

    for model_dirpath in model_dirpaths:
        model_filepath = os.path.join(model_dirpath, 'model.pt')

        p = detector(model_filepath, detpth, calpath=calpath)
        probs.append(p)

        cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
        ys.append(cls)

    roc = roc_auc_score(ys, probs)
    kld = log_loss(ys, probs)
    print('val auc:', roc, 'pre-cal ce:', kld)

def detector(model_filepath, detpth, calpath=None):
    feats = get_jac_feats(model_filepath, nsamples=10000)

    rfmodel = load(detpth)
    p = rfmodel.predict_proba([feats])
    p = p[:, 1]

    if calpath:
        calmodel = load(calpath)
        try:
            p = calmodel.predict(p)
        except:
            # p = calmodel([p])
            p = calmodel.predict_proba([p])[:, 1]
    return p




def det2(train=True, basepath='./'):
    modellist = 'calibration/modelsets/r5_all_trainset.txt'
    # nq=18
    n_estimators=10000
    holdoutratio=0.2
    detname = 'jacdet'

    # dump(model, 'calibration/fitted/' + detname + '_rf.joblib')
    # dump(ir_model, 'calibration/fitted/' + detname + '_ir.joblib')
    # dump(lr_model, 'calibration/fitted/' + detname + '_lr.joblib')


    detpth = 'calibration/fitted/' + detname + '_rf.joblib'
    calpath = 'calibration/fitted/' + detname + '_lr.joblib'

    detpth = os.path.join(basepath, detpth)
    calpath = os.path.join(basepath, calpath)

    if train:
        cal(modellist, detname=detname, holdoutratio=holdoutratio, n_estimators=n_estimators)
    return lambda model_filepath: detector(model_filepath, detpth, calpath=calpath)




def get_jac_feats(model_filepath, ord=np.inf, nsamples=1000, input_scale=1.0, use_mags=False):

    model = torch.load(model_filepath)
    model.parameters()
    model.cuda()
    model.train()

    input_sz = model.parameters().__next__().shape[1]
    # nsamples = 100

    inputs = input_scale*torch.randn([nsamples,1,input_sz],device='cuda')
    if use_mags:
        mags = utils.grad_mag(model, inputs, ord=ord)

        return mags
    else:
        jacobian = utils.compute_jacobian(model, inputs)
        return jacobian.mean(axis=1).reshape(-1)
        # jacobian = jacobian[0]-jacobian[1]
        # return jacobian.mean(axis=0).reshape(-1)




def expcal_jac(modellist, holdoutratio=0.2, nq=18, n_estimators=100, tmppth=None, tmp_overwrite=False, outpth='', nsamples=1000, input_scale=1.0, use_mags=True):
    model_dirpaths = utils.get_modeldirs(modellist, usefile=True)

    x = []
    y = []

    if tmppth is not None and os.path.exists(tmppth):
        with open(tmppth,'rb') as f:
            xsv = pickle.load(f)
    else:
        xsv = {}

    for model_dirpath in model_dirpaths:
        if model_dirpath in xsv:
            feats = xsv[model_dirpath]
        else:
            print('getting feats from', model_dirpath)
            if use_mags:
                feats = [get_jac_feats(os.path.join(model_dirpath, 'model.pt'), ord=1, nsamples=nsamples, input_scale=input_scale),
                         get_jac_feats(os.path.join(model_dirpath, 'model.pt'), ord=2, nsamples=nsamples, input_scale=input_scale),
                         get_jac_feats(os.path.join(model_dirpath, 'model.pt'), ord=np.inf, nsamples=nsamples, input_scale=input_scale)]
            else:
                feats = [get_jac_feats(os.path.join(model_dirpath, 'model.pt'), ord=1, nsamples=nsamples,
                                       input_scale=input_scale, use_mags=use_mags)]

            # feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
            # feats = np.concatenate(feats)
            xsv[model_dirpath] = feats

        import json
        with open(os.path.join(model_dirpath, 'config.json')) as f:
            truth = json.load(f)

        if 'trigger' in truth:
            triggers = [tt["type"]  for tt in truth['trigger']]
        else:
            triggers = []


        if nq:
            if len(feats)==3:
                lcs1, lcs2, lcsi = feats
                qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])
            else:
                qfeats = get_quants(feats[0], nq)
        else:
            # lcs1, lcs2, lcsi = feats
            # qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])
            qfeats = feats[0]

        cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
        arch = utils.get_arch(os.path.join(model_dirpath, 'config.json'))

        if truth['adversarial_training_method'] == 'PGD':
            x.append(qfeats)
            y.append(cls)

    if tmppth is not None and (tmp_overwrite or not os.path.exists(tmppth)):
        with open(tmppth,'wb') as f:
            pickle.dump(xsv, f)
    x = np.stack(x, 0)
    y = np.array(y)

    with open('jacxy.p', 'wb') as f:
        pickle.dump([x,y], f)

    ind = np.arange(len(y))
    np.random.shuffle(ind)

    split = round(len(y) * (1-holdoutratio))

    xtr = x[ind[:split]]
    xv = x[ind[split:]]
    ytr = y[ind[:split]]
    yv = y[ind[split:]]

    for i in range(3):

        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(xtr, ytr)

        pv = model.predict_proba(xv)

        pv = pv[:, 1]

        vroc = roc_auc_score(yv, pv)
        vkld = log_loss(yv, pv)

        ir_model = IsotonicRegression(out_of_bounds='clip')
        pv2 = ir_model.fit_transform(pv, yv)

        vkld2 = log_loss(yv, pv2)
        print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal ce:',vkld2)


def comb_det():
    holdoutratio = 0.2
    n_estimators=1000
    with open('lcxy.p', 'rb') as f:
        [x1,y1] = pickle.load(f)

    with open('jacxy.p', 'rb') as f:
        [x2,y2] = pickle.load(f)

    assert (y1==y2).all()
    x = np.concatenate([x1,x2],axis=1)
    # x = np.concatenate([x2], axis=1)
    # x = np.concatenate([x1], axis=1)
    y=y1

    ind = np.arange(len(y))
    np.random.shuffle(ind)

    split = round(len(y) * (1-holdoutratio))

    xtr = x[ind[:split]]
    xv = x[ind[split:]]
    ytr = y[ind[:split]]
    yv = y[ind[split:]]

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(xtr, ytr)

    pv = model.predict_proba(xv)

    pv = pv[:, 1]

    vroc = roc_auc_score(yv, pv)
    vkld = log_loss(yv, pv)

    ir_model = IsotonicRegression(out_of_bounds='clip')
    pv2 = ir_model.fit_transform(pv, yv)

    vkld2 = log_loss(yv, pv2)
    print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal ce:',vkld2)