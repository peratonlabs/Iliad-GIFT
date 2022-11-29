from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import os
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


class nonneg_logreg_cls(object):

    def __init__(self, args):
        self.args = args
        self.lr_model = LogisticRegression(C=args['C'], max_iter=10000, tol=1e-4)
        self.active_feats = 0

    def predict_proba(self, x):
        x = x * self.active_feats
        self.lr_model.predict_proba(x)

    def fit(self, x, y):
        polarity = self.args['polarity']=='pos'
        done = False

        self.active_feats = np.ones([1, x.shape[1]])
        while not done:
            self.lr_model = LogisticRegression(C=self.args['C'], max_iter=10000, tol=1e-4)
            x = x*self.active_feats
            self.lr_model.fit(x, y)
            if polarity:
                bad_feats = self.lr_model.coef_ < 0 * self.active_feats
            else:
                bad_feats = self.lr_model.coef_ > 0 * self.active_feats
            if bad_feats.sum()>0:
                self.active_feats[bad_feats] = 0
            else:
                done = True


def sv_cls(args, cls_sv, pth, fn):
    os.makedirs(pth, exist_ok=True)
    with open(os.path.join(pth, fn), 'wb') as f:
        pickle.dump([args, cls_sv], f)


def load_cls(pth, args=None):
    with open(pth, 'rb') as f:
        args_sv, cls_sv = pickle.load(f)

    if args is not None:
        assert str(args) == str(args_sv), "detector spec doesn't match saved detector"

    return cls_sv


def logreg_cls(args):
    lr_model = LogisticRegression(C=args['C'], max_iter=10000, tol=1e-4)
    return lr_model

def randforest_cls(args):
    rf_model = RandomForestClassifier(n_estimators=args['n_estimators'], max_iter=10000, tol=1e-4)
    return rf_model

class maxprob_cls(object):

    def __init__(self, args):
        self.args = args
        self.lr_model = LogisticRegression(C=args['C'], max_iter=10000, tol=1e-4)
        self.cals = None

    def predict_proba(self, x):
        x = self.scale_feats(x)
        return self.lr_model.predict_proba(x.max(axis=1, keepdims=True))

    def fit(self, x, y):
        polarity = self.args['polarity'] == 'pos'
        # clip_percentile = self.args['clip_percentile']

        self.cals = []

        for i in range(x.shape[1]):
            cal = IsotonicRegression(increasing=polarity, out_of_bounds='clip')
            cal.fit(x[:, i], y)
            self.cals.append(cal)

        x = self.scale_feats(x)
        self.lr_model.fit(x.max(axis=1, keepdims=True),y)


    def scale_feats(self, x):

        x = [self.cals[i].predict(x[:,i]) for i in range(len(self.cals))]
        x = np.stack(x, axis=1)

        return x


class mahal_dist_cls(object):

    def __init__(self, args, epsilon= 1e-5):
        self.args = args
        self.lr_model = LogisticRegression(C=args['C'], max_iter=10000, tol=1e-4)
        self.xNegMean = None
        self.ZCAMatrix = None
        self.epsilon = epsilon


    def predict_proba(self, x):
        x_w = x - self.xNegMean
        x_w = np.dot(x_w, self.ZCAMatrix) # project X onto the ZCAMatrix
        r = np.linalg.norm(x_w, axis=1).reshape(-1,1)
        xx = np.concatenate([x_w, r], axis = 1)
        return self.lr_model.predict_proba(xx)

    def fit(self, x, y):

        xNeg = x[y==0]
        self.xNegMean = xNeg.mean(axis=0,keepdims=True)
        x_w = x - self.xNegMean

        sigma = np.cov(xNeg, rowvar=False) # [M x M]
        U,S,V = np.linalg.svd(sigma)
        self.ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + self.epsilon)), U.T))  # [M x M]

        x_w = np.dot(x_w, self.ZCAMatrix) # project X onto the ZCAMatrix
        r = np.linalg.norm(x_w,axis=1).reshape(-1,1)
        xx = np.concatenate([x_w,r],axis=1)

        self.lr_model.fit(xx,y)


def run_cv(num_cv_trials, cv_test_prop, cls_fun, cls_args, x, y):


    aucs = []
    ces = []
    for i in range(num_cv_trials):
        xt, xv, yt, yv = train_test_split(x, y, test_size=cv_test_prop, stratify=y)




        # cls_fun = getattr(detection.cls, ner_cls['name'])
        cls_model = cls_fun(cls_args)
        cls_model.fit(xt, yt)

        y_prob = cls_model.predict_proba(xv)[:, 1]
        auc = roc_auc_score(yv, y_prob)
        ce = log_loss(yv, y_prob)

        aucs.append(auc)
        ces.append(ce)
    aucs = np.array(aucs)
    ces = np.array(ces)

    print(f"Cross Validation Mean AUC: {aucs.mean()} and CE: {ces.mean()}")
