# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
# import skimage.io
import random
import torch
import warnings
# import lipsch
from joblib import dump, load
import utils.utils as utils
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import detection.cls

archmap = {'GruLinear':0, 'Linear':1, 'LstmLinear':2}

warnings.filterwarnings("ignore")
TMPFN = 'calibration/fitted/lcdata.p'


def get_svfeats(model_filepath):
    model = torch.load(model_filepath)
    svs = []
    cnt = 0
    for mp in model.parameters():
        # import pdb; pdb.set_trace()
        if len(mp.shape)>=2:
            param = mp.cpu().detach().numpy()

            u, s, vh = np.linalg.svd(param, full_matrices=False)
            # param = param.reshape(param.shape[0], -1)
            svs.append(s)
    
    return svs

def get_feats(model_filepath):
    """
    :param model_filepath:
    :return:
    """
    model = torch.load(model_filepath)
    lcs1 = []
    lcs2 = []
    lcsi = []

    for p in model.parameters():
        param = p.cpu().detach().numpy()
        param = param.reshape(param.shape[0], -1)
        # import pdb; pdb.set_trace()
        lcs1.append(np.linalg.norm(param, ord=1))
        lcs2.append(np.linalg.norm(param, ord=2))
        lcsi.append(np.linalg.norm(param, ord=np.inf))

    Nconst = np.max([np.max(lcs1), np.max(lcs2), np.max(lcsi)])
    return lcs1/Nconst, lcs2/Nconst, lcsi/Nconst

def get_quants(x, n):
    """
    :param x:
    :param n:
    :return:
    """
    q = np.linspace(0, 1, n)
    return np.quantile(x, q)


# def cal(modellist, holdoutratio=0.2, nq=18, n_estimators=100, tmppth=None, tmp_overwrite=False, outpth=''):
def cal(arg_dict, cv_dets, cv_cls):
    holdoutratio=0.2




    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])["cv_dets"]
    nq=metaParameters[0]["nq"]
    # n_estimators=metaParameters[0]["n_estimators"]
    
    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'cv_results')
    os.makedirs(results_dir, exist_ok=True)


    # import pdb;pdb.set_trace()
    
    # model_dirpaths = utils.get_modeldirs(modellist, usefile=True)
    model_dirpaths = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    modelList = os.listdir(model_dirpaths)

    x = []
    y = []

    detstr = str(cv_dets)

    print(len(cv_dets))
    print(cv_dets)

    # tmppth = None
    # if tmppth is not None and os.path.exists(tmppth):
    #     with open(tmppth,'rb') as f:
    #         xsv = pickle.load(f)
    # else:
    #     xsv = {}
    for model_id in modelList:
        print("Current model: ", model_id)
        model_result = None

        model_dirpath = os.path.join(model_dirpaths,model_id)
        # if model_dirpath in xsv:
        #     feats = xsv[model_dirpath]
        # else:
        #     print('getting feats from', model_dirpath)
        #     feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
        #     # feats = np.concatenate(feats)
        #     xsv[model_dirpath] = feats
        res_path = os.path.join(results_dir, model_id + '.p')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                saved_model_result = pickle.load(f)
            if str(saved_model_result["cv_dets"]) == detstr:
                model_result = saved_model_result
            print("loading saved")
            # import pdb; pdb.set_trace()

        if model_result is None:

            # svdfeats = get_svfeats(os.path.join(model_dirpath, 'model.pt'))
            feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
            lcs1, lcs2, lcsi = feats
            qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])

            # import pdb; pdb.set_trace()
            cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
            model_result = {"cv_dets": cv_dets, 'cls': cls, 'features': qfeats}
            with open(res_path, "wb") as f:
                pickle.dump(model_result, f)
            # import pdb; pdb.set_trace()

        
        # arch = utils.get_arch(os.path.join(model_dirpath, 'config.json'))
        # arch = archmap[arch]
        # feats = np.concatenate([get_quants(lcs1, nq),get_quants(lcs2, nq),get_quants(lcsi, nq),[arch]])

        x.append(model_result['features'])
        y.append(model_result['cls'])

    # if tmppth is not None and (tmp_overwrite or not os.path.exists(tmppth)):
    #     with open(tmppth,'wb') as f:
    #         pickle.dump(xsv, f)
    x = np.stack(x, 0)
    y = np.array(y)

    # import pdb; pdb.set_trace()
    n_estimators = 1000
    rfList = []
    lrList = []
    isList = []
    aucList = []
    for _ in range(100):
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

        ir_model = IsotonicRegression(out_of_bounds='clip')
        pv2 = ir_model.fit_transform(pv, yv)

        # pv2 = (pv2 + prior * 0.5) / (1 + prior)
        vkld2 = log_loss(yv, pv2)
        
        indv = np.arange(len(yv))
        np.random.shuffle(indv)

        holdoutratiov = 0.5
        splitv = round(len(yv) * (1-holdoutratiov))

        xtrv = pv[indv[:splitv]]
        xvv = pv[indv[splitv:]]
        ytrv = yv[indv[:splitv]]
        yvv = yv[indv[splitv:]]

        lr_model = LogisticRegression(C=19, max_iter=10000, tol=1e-4)
        lr_model.fit(xtrv.reshape(-1,1), ytrv)
        pv3 = lr_model.predict_proba(xvv.reshape(-1,1))[:,1]
        vkld3 = log_loss(yvv, pv3)
        
        
        
        rfList.append(vkld)
        isList.append(vkld2)
        lrList.append(vkld3)
        aucList.append(vroc)
        #
        print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal (ISO) ce:',vkld2, 'post-cal (LR)', vkld3)
    print(f"mean val Auc: {np.mean(aucList)}, mean RF: {np.mean(rfList)}, mean ISO: {np.mean(isList)}, and mean LR: {np.mean(lrList)}")


    # indv = np.arange(len(yv))
    # np.random.shuffle(indv)

    # holdoutratiov = 0.5
    # splitv = round(len(yv) * (1-holdoutratiov))

    # xtrv = pv[indv[:splitv]]
    # xvv = pv[indv[splitv:]]
    # ytrv = yv[indv[:splitv]]
    # yvv = yv[indv[splitv:]]

    # lr_model = LogisticRegression(C=10, max_iter=10000, tol=1e-4)
    # lr_model.fit(xtrv.reshape(-1,1), ytrv)
    # pv2 = lr_model.predict_proba(xvv.reshape(-1,1))[:,1]
    # vkld2 = log_loss(yvv, pv2)
    # print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal ce:',vkld2)
    # if test_modelist is not None:
    #     x = []
    #     y = []
    #     model_dirpaths = utils.get_modeldirs(test_modelist, usefile=True)
    #     for model_dirpath in model_dirpaths:
    #         if model_dirpath in xsv:
    #             feats = xsv[model_dirpath]
    #         else:
    #             print('getting feats from', model_dirpath)
    #             feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
    #         lcs1, lcs2, lcsi = feats
    #         cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
    #         feats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])
    #
    #         x.append(feats)
    #         y.append(cls)
    #
    #     x = np.stack(x, 0)
    #     y = np.array(y)
    #     p = model.predict_proba(x)
    #     p = p[:, 1]
    #     tstroc = roc_auc_score(y, p)
    #     tstkld = log_loss(y, p)
    #     p2 = ir_model.transform(p)
    #     tstkld2 = log_loss(y, p2)
    #
    #     print('test auc:', tstroc, 'pre-cal ce:', tstkld, 'post-cal ce:', tstkld2)

    # dump(model, outpth)
        # dump(model, detname)
    import pdb; pdb.set_trace()
    dump(model, os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_rf.joblib'))
    dump(ir_model, os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_ir.joblib'))
    # dump(lr_model, 'calibration/fitted/' + outpth + '_lr.joblib')
    # dump(ir_model, output_path + '_ir.joblib')


def detector(model_filepath, rf_path, ir_path, metaParameters):
    nq=metaParameters[0]["nq"]
    feats = get_feats(model_filepath)
    lcs1, lcs2, lcsi = feats
    
    x = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])

    rf_model = load(rf_path)
    ir_model = load(ir_path)
    pv = rf_model.predict_proba([x])[:,1]
    prob = ir_model.transform(pv)

    # import pdb; pdb.set_trace()
    return prob


def det(arg_dict, train=False):

    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname1 = "cv_rf.joblib"
    detname2 = "cv_ir.joblib"
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])["cv_dets"]
    nq=metaParameters[0]["nq"]

    rf_path = os.path.join(learned_parameters_dirpath, detname1) 
    rf_path = os.path.join(basepath, rf_path)
    ir_path = os.path.join(learned_parameters_dirpath, detname2) 
    ir_path = os.path.join(basepath, ir_path)

    if train:
        cal(modellist, outpth=outpth, holdoutratio=holdoutratio, nq=nq, n_estimators=n_estimators)
    return lambda model_filepath: detector(model_filepath, rf_path, ir_path, metaParameters)

def expcal(modellist, holdoutratio=0.2, nq=18, n_estimators=100, tmppth=None, tmp_overwrite=False, outpth=''):
    model_dirpaths = utils.get_modeldirs(modellist, usefile=True)

    # svdpth = 'svd.p'
    # if svdpth is not None and os.path.exists(svdpth):
    #     with open(svdpth,'rb') as f:
    #         svdfeats = pickle.load(f)
    # else:
    #     svdfeats = {}
    # for model_dirpath in model_dirpaths:
    #     if model_dirpath in svdfeats:
    #         feats = svdfeats[model_dirpath]
    #     else:
    #         print('getting feats from', model_dirpath)
    #         feats = get_svfeats(os.path.join(model_dirpath, 'model.pt'))
    #         # feats = np.concatenate(feats)
    #         svdfeats[model_dirpath] = feats
    # if svdpth is not None and  not os.path.exists(svdpth):
    #     with open(svdpth,'wb') as f:
    #         pickle.dump(svdfeats, f)

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
            feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
            # feats = np.concatenate(feats)
            xsv[model_dirpath] = feats

        import json
        with open(os.path.join(model_dirpath, 'config.json')) as f:
            truth = json.load(f)

        if 'trigger' in truth:
            triggers = [tt["type"]  for tt in truth['trigger']]
        else:
            triggers = []



        # this_svdfeats = svdfeats[model_dirpath]
        #
        # this_svdfeats = this_svdfeats[0]
        # if this_svdfeats.shape[0] != 768:
        #     tmp = np.zeros(768)
        #     tmp[:this_svdfeats.shape[0]] += this_svdfeats
        #     this_svdfeats = tmp


        # print(this_svdfeats[1].shape)

        # this_svdfeats = np.concatenate(this_svdfeats)

        # this_svdfeats = get_quants(this_svdfeats, 200)


        # this_svdfeats

        lcs1, lcs2, lcsi = feats

        qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])

        # lcs1 += [0 for i in range(18-len(lcs1))]
        # lcs2 += [0 for i in range(18-len(lcs2))]
        # lcsi += [0 for i in range(18-len(lcsi))]
        # cfeats = np.concatenate([lcs1, lcs2, lcsi])


        cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
        arch = utils.get_arch(os.path.join(model_dirpath, 'config.json'))
        # arch = archmap[arch]
        # feats = np.concatenate([get_quants(lcs1, nq),get_quants(lcs2, nq),get_quants(lcsi, nq),[arch]])

        # if
        if truth['adversarial_training_method'] == 'PGD':
        # if truth['embedding'] == 'GPT-2':
            # if len(lcs1)==18:
            # 'GruLinear': 0, 'Linear': 1, 'LstmLinear': 2
            # if arch == 'Linear':
            # x.append(cfeats)
            x.append(qfeats)
            # x.append(this_svdfeats)
            # x.append(np.concatenate([qfeats, this_svdfeats]))
            y.append(cls)

    if tmppth is not None and (tmp_overwrite or not os.path.exists(tmppth)):
        with open(tmppth,'wb') as f:
            pickle.dump(xsv, f)
    x = np.stack(x, 0)
    y = np.array(y)

    with open('lcxy.p', 'wb') as f:
        pickle.dump([x,y], f)

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

    ir_model = IsotonicRegression(out_of_bounds='clip')
    pv2 = ir_model.fit_transform(pv, yv)

    # pv2 = (pv2 + prior * 0.5) / (1 + prior)
    vkld2 = log_loss(yv, pv2)
    #
    print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal ce:',vkld2)

    # if test_modelist is not None:
    #     x = []
    #     y = []
    #     model_dirpaths = utils.get_modeldirs(test_modelist, usefile=True)
    #     for model_dirpath in model_dirpaths:
    #         if model_dirpath in xsv:
    #             feats = xsv[model_dirpath]
    #         else:
    #             print('getting feats from', model_dirpath)
    #             feats = get_feats(os.path.join(model_dirpath, 'model.pt'))
    #         lcs1, lcs2, lcsi = feats
    #         cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
    #         feats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)])
    #
    #         x.append(feats)
    #         y.append(cls)
    #
    #     x = np.stack(x, 0)
    #     y = np.array(y)
    #     p = model.predict_proba(x)
    #     p = p[:, 1]
    #     tstroc = roc_auc_score(y, p)
    #     tstkld = log_loss(y, p)
    #     p2 = ir_model.transform(p)
    #     tstkld2 = log_loss(y, p2)
    #
    #     print('test auc:', tstroc, 'pre-cal ce:', tstkld, 'post-cal ce:', tstkld2)

    # dump(model, outpth)
    # dump(ir_model, output_path + '_ir.joblib')


def get_jac_feats(model_filepath, ord=np.inf, nsamples=1000, input_scale=1.0, use_mags=True):

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

    for i in range(30):

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