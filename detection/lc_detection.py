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
from sklearn.ensemble import GradientBoostingClassifier


import detection.cls
# import models
archmap = {'GruLinear':0, 'Linear':1, 'LstmLinear':2}

warnings.filterwarnings("ignore")
TMPFN = 'calibration/fitted/lcdata.p'

def get_linfeats(model_filepath):
    model = torch.load(model_filepath)
    if type(model) is torchvision.models.resnet.ResNet:
        modeltype=0
        pinds = [133, 148, 130, 121, 139]
    elif type(model) is torchvision.models.mobilenetv2.MobileNetV2:
        modeltype = 1
        pinds = [148, 130, 121, 139]
    elif type(model) is timm.models.vision_transformer.VisionTransformer:
        modeltype = 2
        pinds = [61,73]

    linfeats = []
    ps = [mp for mp in model.parameters() ]

    for pind in pinds:
        
        param = ps[pind].cpu().detach().numpy()
        param = param.reshape(-1)
            
        linfeats.append(param)
    linfeats = np.concatenate(linfeats)
    return linfeats, modeltype



def get_qqfeats(model_filepath, n=20):
    model = torch.load(model_filepath)
    if type(model) is torchvision.models.resnet.ResNet:
        modeltype=0
    elif type(model) is torchvision.models.mobilenetv2.MobileNetV2:
        modeltype = 1
    elif type(model) is timm.models.vision_transformer.VisionTransformer:
        modeltype = 2

    svs = []
    cnt = 0
    ps = [mp for mp in model.parameters() if len(mp.shape)>=2]

    for mp in ps:
        if len(mp.shape)>=2:
            param = mp.cpu().detach().numpy()
            param = param.reshape(-1)
            
            svs.append(get_quants(param , n))

    return svs, modeltype

def get_svfeats(model_filepath):
    model = torch.load(model_filepath)
    if type(model) is torchvision.models.resnet.ResNet:
        modeltype=0
    elif type(model) is torchvision.models.mobilenetv2.MobileNetV2:
        modeltype = 1
    elif type(model) is timm.models.vision_transformer.VisionTransformer:
        modeltype = 2



    svs = []
    cnt = 0
    ps = [mp for mp in model.parameters() if len(mp.shape)>=2]
    #ps = ps[-3:-1]
    ps = ps[-3:]
    #ps = ps[-1:]
    #ps = ps[-2:-1]
    #ps = ps[-3:-2]
    #ps = ps[-10:]

    for mp in ps:
        # import pdb; pdb.set_trace()
        if len(mp.shape)>=2:
            #print(mp.shape)
            param = mp.cpu().detach().numpy()
            param = param.reshape(param.shape[0], -1)


            u, s, vh = np.linalg.svd(param, full_matrices=False)
            # param = param.reshape(param.shape[0], -1)
            svs.append(s[:5])
            svs.append(s[-5:])
            #print(mp.shape, s.shape)

    #print(svs[-3].shape)
    return svs, modeltype


import torchvision
from torchvision.models import resnet50, mobilenet_v2
import timm


def get_feats(model_filepath, gift_basepath):
    """
    :param model_filepath:
    :return:
    """
    model = torch.load(model_filepath)
    # print(type(model))
    if type(model) is torchvision.models.resnet.ResNet:
        refmodel = resnet50()
        missing_keys, unexpected_keys = refmodel.load_state_dict(torch.load(os.path.join(gift_basepath, 'resnet50_V2.pth')))
        modeltype=0
    elif type(model) is torchvision.models.mobilenetv2.MobileNetV2:
        refmodel = mobilenet_v2()
        missing_keys, unexpected_keys = refmodel.load_state_dict(torch.load(os.path.join(gift_basepath, 'mobilenet_V2.pth')))
        modeltype = 1
    elif type(model) is timm.models.vision_transformer.VisionTransformer:
        refmodel = timm.create_model('vit_base_patch16_224', pretrained=False)
        refmodel.load_state_dict(torch.load(os.path.join(gift_basepath, 'vit.pt')))
        modeltype = 2

    else:
        refmodel = None
    

    lcs1 = []
    lcs2 = []
    lcsi = []
    lcs1b = []
    lcs2b = []
    lcsib = []


    ii=0
    for p, refp in zip(model.parameters(), refmodel.parameters()):
        # print(type(model))
        # print("Origin: ", p.shape)
        
        # print("reference: ", refp.shape)
        # print()
        # print("\n")
        
        if p.shape==refp.shape:
            param = (p-refp).cpu().detach().numpy()
            if len(param.shape) > 0:
                param = param.reshape(-1)
                lcs1b.append(np.linalg.norm(param, ord=1))
                lcs2b.append(np.linalg.norm(param, ord=2))
                lcsib.append(np.linalg.norm(param, ord=np.inf))
        ii+=1

        # input()

    iii = 0
    for p in model.parameters():
        param = p.cpu().detach().numpy()
        if len(param.shape)>0:
            param = param.reshape(-1)
            lcs1.append(np.linalg.norm(param, ord=1))
            lcs2.append(np.linalg.norm(param, ord=2))
            lcsi.append(np.linalg.norm(param, ord=np.inf))
        iii += 1

    return lcs1/np.max(lcs1), lcs2/np.max(lcs2), lcsi/np.max(lcsi), lcs1b/np.max(lcs1b), lcs2b/np.max(lcs2b), lcsib/np.max(lcsib), modeltype

def get_quants(x, n):
    """
    :param x:
    :param n:
    :return:
    """
    q = np.linspace(0, 1, n)
    return np.quantile(x, q)


# def cal(modellist, holdoutratio=0.2, nq=18, n_estimators=100, tmppth=None, tmp_overwrite=False, outpth=''):
def cal(arg_dict, metaParameters):
    holdoutratio=0.1
    

    modelsplit = metaParameters["modelsplit"]
    max_depth=metaParameters["max_depth"]

    nq=metaParameters["nq"]
    n_estimators = metaParameters["n_estimators"]
    # n_estimators=metaParameters[0]["n_estimators"]

    gift_basepath = arg_dict['gift_basepath']
    num_cv_trials = arg_dict['num_cv_trials']

    scratch_dirpath = arg_dict['scratch_dirpath']
    results_dir = os.path.join(scratch_dirpath, 'cv_results')
    os.makedirs(results_dir, exist_ok=True)



    
    # model_dirpaths = utils.get_modeldirs(modellist, usefile=True)
    model_dirpaths = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    modelList = os.listdir(model_dirpaths)
    modelList.sort()

    x = []
    y = []

    detstr = 'str'
    cv_dets = [detstr]

    # detstr = str(cv_dets)
    #
    # print(len(cv_dets))
    # print(cv_dets)

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
            #print(str(saved_model_result["cv_dets"]))
            #print(detstr)
            #if str(saved_model_result["cv_dets"]) == detstr:
            model_result = saved_model_result
            print("loading saved")
            # import pdb; pdb.set_trace()

        if model_result is None:

            # svdfeats = get_svfeats(os.path.join(model_dirpath, 'model.pt'))
            
            #feats = get_feats(os.path.join(model_dirpath, 'model.pt'), gift_basepath)
            feats, modeltype = get_linfeats(os.path.join(model_dirpath, 'model.pt'))
            qfeats = np.concatenate([[modeltype], feats])


            # lcs1, lcs2, lcsi  = feats
            # qfeats = np.concatenate([lcs1, lcs2, lcsi])
            #lcs1, lcs2, lcsi, lcs1b, lcs2b, lcsib, modeltype  = feats
            #qfeats = np.concatenate([[modeltype], lcs1, lcs2, lcsi, lcs1b, lcs2b, lcsib])

            #svs, _ = get_svfeats(os.path.join(model_dirpath, 'model.pt'))
            #qqfeats, _ = get_qqfeats(os.path.join(model_dirpath, 'model.pt'))
            #svs = np.concatenate(svs)
            #qqfeats = np.concatenate(qqfeats)

            #print(qfeats.shape )
            #print(svs.shape )
            #print(qqfeats.shape )





            #qfeats = np.concatenate([qfeats , svs, qqfeats])
            print(qfeats.shape )
            # import pdb; pdb.set_trace()
            #maxfeats = 2071
            #maxfeats = 961
            #if qfeats.shape[0] < maxfeats:
            #    qfeats = np.concatenate([qfeats, np.zeros(maxfeats - qfeats.shape[0])])



            # qfeats = np.concatenate([get_quants(lcs1, nq), get_quants(lcs2, nq), get_quants(lcsi, nq)] + [[modeltype]])
            print("Shaoe", qfeats.shape)
            # import pdb; pdb.set_trace()

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
    

    maxfeats  = np.max([len(xx) for xx in x])
    for i in range(len(x)):
        x[i] = np.concatenate([x[i], np.zeros(maxfeats - x[i].shape[0])])
        

    x = np.stack(x, 0)
    y = np.array(y)

    # import pdb; pdb.set_trace()

    rocList = []
    rfList = []
    lrList = []
    isList = []

    rf_scores = []
    truths = []
    modeltypes = []

    numSample = num_cv_trials
    for _ in range(numSample):
        ind = np.arange(len(y))
        np.random.shuffle(ind)


        split = round(len(y) * (1-holdoutratio))

        xtr = x[ind[:split]]
        xv = x[ind[split:]]
        ytr = y[ind[:split]]
        yv = y[ind[split:]]


        if modelsplit:

            xtr0 = xtr[xtr[:, 0] == 0]
            ytr0 = ytr[xtr[:, 0] == 0]
            xtr1 = xtr[xtr[:, 0] == 1]
            ytr1 = ytr[xtr[:, 0] == 1]
            xtr2 = xtr[xtr[:, 0] == 2]
            ytr2 = ytr[xtr[:, 0] == 2]

            xv0 = xv[xv[:, 0] == 0]
            yv0 = yv[xv[:, 0] == 0]
            xv1 = xv[xv[:, 0] == 1]
            yv1 = yv[xv[:, 0] == 1]
            xv2 = xv[xv[:, 0] == 2]
            yv2 = yv[xv[:, 0] == 2]

            #model0 = RandomForestClassifier(n_estimators=n_estimators)
            #model1 = RandomForestClassifier(n_estimators=n_estimators)
            #model2 = RandomForestClassifier(n_estimators=n_estimators)
            #model0 = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
            #model1 = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
            #model2 = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model0 = LogisticRegression(max_iter=1000, C=3)
            model1 = LogisticRegression(max_iter=1000, C=3)
            model2 = LogisticRegression(max_iter=1000, C=3)



            model0.fit(xtr0, ytr0)
            model1.fit(xtr1, ytr1)
            model2.fit(xtr2, ytr2)

            pv0 = model0.predict_proba(xv0)
            pv0 = pv0[:, 1]
            pv1 = model1.predict_proba(xv1)
            pv1 = pv1[:, 1]
            pv2 = model2.predict_proba(xv2)
            pv2 = pv2[:, 1]

            pv = np.concatenate([pv0, pv1,pv2])
            yv = np.concatenate([yv0, yv1, yv2])
            xv = np.concatenate([xv0, xv1, xv2])
  

            try:
                print('modelsplit auc:', roc_auc_score(yv0, pv0), roc_auc_score(yv1, pv1),  roc_auc_score(yv2, pv2), 'ce', log_loss(yv0, pv0), log_loss(yv1, pv1), log_loss(yv2, pv2))
            except:
                pass

        else:
            model = LogisticRegression(max_iter=1000, C=3)
            model.fit(xtr, ytr)
            pv = model.predict_proba(xv)
            pv = pv[:, 1]

        vroc = roc_auc_score(yv, pv)
        vkld = log_loss(yv, pv)

        rf_scores.append(pv)
        truths.append(yv)
        modeltypes.append(xv[:,0])


        vkld2=0
        vkld3=0
        
        rfList.append(vkld)
        isList.append(vkld2)
        lrList.append(vkld3)
        rocList.append(vroc)
        #
        print('val auc:', vroc, 'pre-cal ce:', vkld, 'post-cal (ISO) ce:',vkld2, 'post-cal (LR)', vkld3)


    rf_scores = np.array(rf_scores)
    truths = np.array(truths)
    modeltypes = np.array(modeltypes)

    ISOce_scores = []
    LRce_scores = []
    for _ in range(numSample):
        ind = np.arange(len(rf_scores))
        np.random.shuffle(ind)
        split = round(len(rf_scores) * (1-holdoutratio))

        ptr = np.concatenate(rf_scores[ind[:split]])
        ptst = np.concatenate(rf_scores[ind[split:]])
        ytr = np.concatenate(truths[ind[:split]])
        ytst = np.concatenate(truths[ind[split:]])

        if modelsplit:
            mt_tr = np.concatenate(modeltypes[ind[:split]])
            mt_tst = np.concatenate(modeltypes[ind[split:]])

            ptr0 = ptr[mt_tr == 0]
            ptr1 = ptr[mt_tr == 1]
            ptr2 = ptr[mt_tr == 2]
            ytr0 = ytr[mt_tr == 0]
            ytr1 = ytr[mt_tr == 1]
            ytr2 = ytr[mt_tr == 2]

            ptst0 = ptst[mt_tst == 0]
            ptst1 = ptst[mt_tst == 1]
            ptst2 = ptst[mt_tst == 2]
            ytst0 = ytst[mt_tst == 0]
            ytst1 = ytst[mt_tst == 1]
            ytst2 = ytst[mt_tst == 2]

            ir_model0 = IsotonicRegression(out_of_bounds='clip')
            ir_model0.fit(ptr0, ytr0)
            p2tst0 = ir_model0.transform(ptst0)
            p2tst0 = np.clip(p2tst0, 0.01, 0.99)

            ir_model1 = IsotonicRegression(out_of_bounds='clip')
            ir_model1.fit(ptr1, ytr1)
            p2tst1 = ir_model1.transform(ptst1)
            p2tst1 = np.clip(p2tst1, 0.01, 0.99)

            ir_model2 = IsotonicRegression(out_of_bounds='clip')
            ir_model2.fit(ptr2, ytr2)
            p2tst2 = ir_model2.transform(ptst2)
            p2tst2 = np.clip(p2tst2, 0.01, 0.99)
            ytst = np.concatenate([ytst0,ytst1,ytst2])
            p2tst = np.concatenate([p2tst0,p2tst1, p2tst2])

            ISOce_scores.append(log_loss(ytst, p2tst))
        else:

            ir_model = IsotonicRegression(out_of_bounds='clip')
            ir_model.fit(ptr, ytr)
            p2tst = ir_model.transform(ptst)
            # pv2 = (pv2 + prior * 0.5) / (1 + prior)
            p2tst = np.clip(p2tst, 0.01, 0.99)
            # vkld2 = log_loss(ytst, p2tst)
            ISOce_scores.append(log_loss(ytst, p2tst))

        lr_model = LogisticRegression(C=100, max_iter=10000, tol=1e-4)
        lr_model.fit(ptr.reshape(-1,1), ytr)
        p2tst = lr_model.predict_proba(ptst.reshape(-1,1))[:,1]
        # vkld3 = log_loss(ytst, p2tst)
        LRce_scores.append(log_loss(ytst, p2tst))



    print(f"Val Auc: {np.mean(rocList)}, mean RF: {np.mean(rfList)}, mean ISO: {np.mean(isList)}, and mean LR: {np.mean(lrList)}")
    print('new ISO CE', np.mean(ISOce_scores),'new LR CE', np.mean(LRce_scores))

    print("modelsplit", modelsplit)

    if modelsplit:
        x0 = x[x[:, 0] == 0]
        y0 = y[x[:, 0] == 0]
        x1 = x[x[:, 0] == 1]
        y1 = y[x[:, 0] == 1]
        x2 = x[x[:, 0] == 2]
        y2 = y[x[:, 0] == 2]

        model0 = LogisticRegression(max_iter=1000, C=3)
        model1 = LogisticRegression(max_iter=1000, C=3)
        model2 = LogisticRegression(max_iter=1000, C=3)

        model0.fit(x0, y0)
        model1.fit(x1, y1)
        model2.fit(x2, y2)

        mt = np.concatenate(modeltypes)
        rf_scores = np.concatenate(rf_scores)
        y = np.concatenate(truths)

        rf_scores0 = rf_scores[mt == 0]
        rf_scores1 = rf_scores[mt == 1]
        rf_scores2 = rf_scores[mt == 2]
        y0 = y[mt == 0]
        y1 = y[mt == 1]
        y2 = y[mt == 2]


        ir_model0 = IsotonicRegression(out_of_bounds='clip')
        ir_model1 = IsotonicRegression(out_of_bounds='clip')
        ir_model2 = IsotonicRegression(out_of_bounds='clip')

        ir_model0.fit(rf_scores0, y0)
        ir_model1.fit(rf_scores1, y1)
        ir_model2.fit(rf_scores2, y2)

        dump([model0, model1, model2], os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_rf.joblib'))
        dump([ir_model0, ir_model1, ir_model2], os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_ir.joblib'))

    else:
        model = LogisticRegression(max_iter=1000, C=3)
        model.fit(x, y)




        ir_model = IsotonicRegression(out_of_bounds='clip')
        ir_model.fit(np.concatenate(rf_scores), np.concatenate(truths))

    # dump(model, outpth)
        # dump(model, detname)
    #import pdb; pdb.set_trace()
        dump(model, os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_rf.joblib'))
        dump(ir_model, os.path.join(arg_dict['learned_parameters_dirpath'], 'cv_ir.joblib'))
    # dump(lr_model, 'calibration/fitted/' + outpth + '_lr.joblib')
    # dump(ir_model, output_path + '_ir.joblib')


def detector(model_filepath, rf_path, ir_path, metaParameters, gift_basepath):
    nq=metaParameters["nq"]
    modelsplit =metaParameters["modelsplit"]


    feats, modeltype = get_linfeats(model_filepath)
    qfeats = np.concatenate([[modeltype], feats])
    #feats = get_feats(model_filepath, gift_basepath)

    #lcs1, lcs2, lcsi, lcs1b, lcs2b, lcsib, modeltype  = feats
    #qfeats = np.concatenate([[modeltype], lcs1, lcs2, lcsi, lcs1b, lcs2b, lcsib])

    #svs, _ = get_svfeats(model_filepath)
    #qqfeats, _ = get_qqfeats(model_filepath)
    #svs = np.concatenate(svs)
    #qqfeats = np.concatenate(qqfeats)

    #qfeats = np.concatenate([qfeats , svs, qqfeats])


    if modelsplit:
        rf_model0, rf_model1, rf_model2 = load(rf_path)
        ir_model0, ir_model1, ir_model2 = load(ir_path)

        if modeltype==0:
            maxfeats = rf_model0.n_features_in_
            if qfeats.shape[0] < maxfeats:
                qfeats = np.concatenate([qfeats, np.zeros(maxfeats - qfeats.shape[0])])
            pv = rf_model0.predict_proba([qfeats])[:, 1]
            prob = ir_model0.transform(pv)
        elif modeltype==1:
            maxfeats = rf_model1.n_features_in_
            if qfeats.shape[0] < maxfeats:
                qfeats = np.concatenate([qfeats, np.zeros(maxfeats - qfeats.shape[0])])

            pv = rf_model1.predict_proba([qfeats])[:, 1]
            prob = ir_model1.transform(pv)
        else:
            maxfeats = rf_model1.n_features_in_
            if qfeats.shape[0] < maxfeats:
                qfeats = np.concatenate([qfeats, np.zeros(maxfeats - qfeats.shape[0])])

            pv = rf_model2.predict_proba([qfeats])[:, 1]
            prob = ir_model2.transform(pv)
    else:

        rf_model = load(rf_path)
        ir_model = load(ir_path)
        maxfeats = rf_model.n_features_in_
        if qfeats.shape[0] < maxfeats:
            qfeats = np.concatenate([qfeats, np.zeros(maxfeats - qfeats.shape[0])])

        pv = rf_model.predict_proba([qfeats])[:,1]
        prob = ir_model.transform(pv)

    # import pdb; pdb.set_trace()
    #return prob
    return pv


def det(arg_dict, train=False):

    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname1 = "cv_rf.joblib"
    detname2 = "cv_ir.joblib"
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])
    nq=metaParameters["nq"]

    rf_path = os.path.join(learned_parameters_dirpath, detname1) 
    rf_path = os.path.join(basepath, rf_path)
    ir_path = os.path.join(learned_parameters_dirpath, detname2) 
    ir_path = os.path.join(basepath, ir_path)

    if train:
        cal(modellist, outpth=outpth, holdoutratio=holdoutratio, nq=nq, n_estimators=n_estimators, gift_basepath=arg_dict["gift_basepath"])
    return lambda model_filepath: detector(model_filepath, rf_path, ir_path, metaParameters, basepath)

