

import torch
import numpy as np
# import lightgbm as lgb
import os
import pickle
from utils import utils
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss


def get_arch(model_filepath):
    model = torch.load(model_filepath)
    cls = str(type(model))
    return cls


def get_archmap(model_filepaths):
    arch_list = [get_arch(model_filepath) for model_filepath in model_filepaths]
    arch_map = {}
    for i, arch in enumerate(arch_list):
        if arch not in arch_map:
            arch_map[arch] = []
        # arch_map[arch].append([model_filepaths[i], cls[i]])
        arch_map[arch].append(i)
    return arch_map



def arch_train(arch_fns, arch_classes, cfg_dict):
    nfeats = cfg_dict['nfeats']
    cls_type = cfg_dict['cls_type']
    param_batch_sz = cfg_dict['param_batch_sz']

    weight_mapping = select_feats(arch_fns, arch_classes, nfeats, param_batch_sz=param_batch_sz)
    x = [get_mapped_weights(fn, weight_mapping) for fn in arch_fns]
    x = np.stack(x)

    ux =x.mean(axis=0)
    stdx =x.std(axis=0)
    x = (x-ux)/stdx
    xstats = [ux, stdx]

    if cls_type == 'LogisticRegression':
        classifier = LogisticRegression(max_iter=1000, C=cfg_dict['C'])
    elif cls_type == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=10000)
        # classifier = lgb.LGBMClassifier(boosting_type='rf', n_estimators=10000)

    elif cls_type == 'GB':
        # classifier = lgb.LGBMClassifier(boosting_type='goss', n_estimators=500,max_depth=1)
        # classifier = lgb.LGBMClassifier(boosting_type='goss', n_estimators=500,max_depth=2)
        # classifier = lgb.LGBMClassifier(n_estimators=500,max_depth=2)
        classifier = lgb.LGBMClassifier(n_estimators=1000,max_depth=2)


    elif cls_type == 'calavg':
        classifier = CalAvgCls()
        # classifier = CalAvgCls2()

    else:
        assert False, 'bad classifier type'
    classifier.fit(x, arch_classes)

    return weight_mapping, classifier, xstats


def get_mets(y,x,thr):
    pred = x >= thr
    tp = torch.logical_and(pred == 1, y == 1).sum(axis=0)
    fp = torch.logical_and(pred == 1, y == 0).sum(axis=0)
    tn = torch.logical_and(pred == 0, y == 0).sum(axis=0)
    fn = torch.logical_and(pred == 0, y == 1).sum(axis=0)

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return [fpr, tpr]

def get_auc(x,y):
    # y = np.tile(y,x.shape[1])
    y = y.reshape(-1,1)
    # t = time.time()
    xsorted = x.copy()
    xsorted.sort(axis=0)
    # print('sorting', time.time()- t)

    x = torch.tensor(x).cuda()
    y = torch.tensor(y).cuda()
    # xsorted = torch.tensor(xsorted).cuda()


    tprs = []
    fprs = []
    # t = time.time()
    for i in range(x.shape[0]):
        # fpr, tpr = get_mets(y, x, xsorted[i:i+1])
        fpr, tpr = get_mets(y, x, torch.tensor(xsorted[i:i+1]).cuda())
        tprs.append(tpr.cpu().detach().numpy())
        fprs.append(fpr.cpu().detach().numpy())
    # print('get_mets', time.time()- t)

    tprs = np.stack(tprs)
    fprs = np.stack(fprs)

    tprs = np.stack([tprs[:-1] , tprs[1:]],axis=1)
    fprs = np.stack([fprs[:-1] , fprs[1:]],axis=1)

    # t = time.time()
    auc = 1- (np.sum(np.trapz(tprs, fprs, axis=1), axis=0 )+ 1)
    # print('trap',time.time() - t)

    return auc


def get_corr(x, y):

    y = y.reshape(-1,1)
    ux = x.mean(axis=0).reshape(1,-1)
    uy = y.mean(axis=0).reshape(1,-1)

    stdx = x.std(axis=0).reshape(1,-1)   #* (y.shape[0])/(y.shape[0]-1)
    stdy = y.std(axis=0).reshape(1,-1)   #* (y.shape[0])/(y.shape[0]-1)

    cov = (x-ux) * (y-uy)
    cov = cov.sum(axis=0)/(y.shape[0]-1)

    corr = cov/(stdx*stdy* (y.shape[0])/(y.shape[0]-1))

    return corr


def get_mapped_weights(model_filepath, weight_mapping):
    model = torch.load(model_filepath)
    ps = [mp for mp in model.parameters()]
    mapped_weights = []
    for i in range(len(weight_mapping)):
        param = ps[i].cpu().detach().numpy()
        param = param.reshape(-1)
        mapped_weights.append(param[weight_mapping[i]])
    mapped_weights = np.concatenate(mapped_weights)
    return mapped_weights



def get_param(models, ind):

    num_weights = None
    vectors = []
    for model in models:
        if not isinstance(model, torch.nn.Module):
            model = torch.load(model)

        ps = [mp for mp in model.parameters()]

        if ind >= len(ps):
            return None

        param = ps[ind].cpu().detach().numpy()
        param = param.reshape(-1)

        if num_weights is None:
            num_weights = param.shape[0]
        else:
            if num_weights != param.shape[0]:
                return None
        vectors.append(param)

    return np.stack(vectors)


def get_params(models, start_ind, num_params):

    if start_ind==150:
        pp=1
    output_ps = []
    # output_szs = []
    for model in models:
        if not isinstance(model, torch.nn.Module):
            model = torch.load(model)

        ps = [mp for mp in model.parameters()]
        ps = ps[start_ind:start_ind+num_params]
        ps = [p.cpu().detach().numpy().reshape(-1) for p in ps]

        if len(output_ps)==0:
            output_ps = [[p] for p in ps]
        else:
            for i,p in enumerate(ps):
                if output_ps[i][0].shape[0] == p.shape[0]:
                    output_ps[i].append(p)
                else:
                    num_params = i
                    output_ps = output_ps[:i]
                    break

    output_ps = [np.stack(vectors) for vectors in output_ps]

    return output_ps


def detect(fn, weight_mapping, classifier, xstats):
    x = [get_mapped_weights(fn, weight_mapping)]

    ux, stdx = xstats
    x = (x - ux) / stdx
    p = classifier.predict_proba(x)[:, 1]
    return p

def cv_train(model_filepaths, cls, cfg_dict, num_cvs=10, holdout_ratio=0.1):
    arch_map = get_archmap(model_filepaths)
    arch_weight_mappings = {}
    arch_classifiers = {}
    arch_xstats = {}

    for arch, arch_inds in arch_map.items():
        print('starting arch', arch)

        arch_fns = np.array([model_filepaths[i] for i in arch_inds])
        arch_classes = np.array([cls[i] for i in arch_inds])

        ns = arch_classes.shape[0]
        inds = np.arange(ns)
        split_ind = round((1-holdout_ratio)*ns)

        cvcal_scores = []
        truths = []
        for i in range(num_cvs):
            np.random.shuffle(inds)
            trinds = inds[:split_ind]
            vinds = inds[split_ind:]

            tr_fns = arch_fns[trinds]
            tr_cls = arch_classes[trinds]
            v_fns = arch_fns[vinds]
            v_cls = arch_classes[vinds]

            weight_mapping, classifier, xstats = arch_train(tr_fns, tr_cls, cfg_dict)



            # xv = [get_mapped_weights(fn, weight_mapping) for fn in v_fns]

            pv = [detect(fn, weight_mapping, classifier, xstats) for fn in v_fns]


            # pv = classifier.predict_proba(xv)[:, 1]
            print(roc_auc_score(v_cls, pv), log_loss(v_cls, pv))
            cvcal_scores.append(pv)
            truths.append(v_cls)



        weight_mapping, classifier, xstats = arch_train(arch_fns, arch_classes, cfg_dict)
        arch_weight_mappings[arch] = weight_mapping
        arch_classifiers[arch] = classifier
        arch_xstats[arch] = xstats
    return arch_weight_mappings, arch_classifiers, arch_xstats


def select_feats(model_fns, labels, nfeats, criterion='auc', param_batch_sz=10):
    ind = 0
    aucs = []
    while True:
        # print('starting param', ind)
        xs = get_params(model_fns, ind, param_batch_sz)

        for x in xs:
            if criterion == 'auc':
                this_aucs = np.abs(get_auc(x, labels).astype(np.float64) - 0.5)
            elif criterion == 'corr':
                this_aucs = np.abs(get_auc(x, labels).astype(np.float64) - 0.5)
            else:
                assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
            this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)
            aucs.append(this_aucs)

        if len(xs) < param_batch_sz:
            break

        ind += param_batch_sz

    aucscopy = np.concatenate(aucs)
    aucscopy.sort()
    thr = aucscopy[-nfeats]

    # featmap = aucs >= thr

    weight_mapping = []
    for auc in aucs:
        weight_mapping.append(auc >= thr)

    return weight_mapping
    # x = [get_mapped_weights(fn, weight_mapping) for fn in arch_fns]
    #
    # x = np.stack(x)



class CalAvgCls():

    def fit(self, x, y):
        self.ux = x.mean(axis=0)
        self.stdx = x.std(axis=0)
        corr = get_corr(x, y)
        self.sign = np.sign(corr)

    def predict_proba(self, x):
        p = (self.sign * (x - self.ux) / self.stdx).mean(axis=1)
        p = 1/(1 + np.exp(-p))
        return np.stack([1-p,p], axis=1)


class CalAvgCls2():

    def fit(self, x, y):
        self.ux = x.mean(axis=0)
        self.stdx = x.std(axis=0)
        corr = get_corr(x, y)
        self.sign = np.sign(corr)
        self.y = y
        self.ref_feats = (self.sign * (x - self.ux) / self.stdx)
        # self.ref_feats.sort()
        # self.probs = np.arange(len(self.ref_feats)) / float(len(self.ref_feats))
        # self.cdfs = [ECDF(self.ref_feats[:,i]) for i in range(self.ref_feats.shape[1])]


    def predict_proba(self, x):
        feats = (self.sign * (x - self.ux) / self.stdx)

        # y = np.arange(len(x)) / float(len(x))

        # self.ref_feats

        # p = [f, ref in ]
        calfeats = 0.0*feats + 1.0

        for row in range(feats.shape[0]):
            for col in range(feats.shape[1]):
                # inds = np.where(self.ref_feats>feats[row, col])[1]
                ps = self.y[self.ref_feats[:,col]>feats[row, col]]
                if ps.shape[0]>0:
                    calfeats[row,col] = ps.mean()

        p = calfeats.mean(axis=1)
        # pp=1
        # p = (self.sign * (x - self.ux) / self.stdx).mean(axis=1)
        # p = 1/(1 + np.exp(-p))
        return np.stack([1-p,p], axis=1)

def demo(base_path='/media/ssd2/trojai/r11/models'):
    import os
    from utils import utils
    modeldirs = os.listdir(base_path)
    
    # modeldirs = modeldirs[:40]

    model_filepaths = [os.path.join(base_path, modeldir, 'model.pt') for modeldir in modeldirs]

    cls = [utils.get_class(os.path.join(base_path, modeldir, 'config.json')) for modeldir in modeldirs]
    mappings, classifiers, xstats = cv_train(model_filepaths, cls, cfg_dict={'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 0.03})
    import pdb; pdb.set_trace()
    dump([mappings, classifiers, xstats], os.path.join(arg_dict['learned_parameters_dirpath'], 'wa_lr.joblib'))
    # import pdb; pdb.set_trace()



