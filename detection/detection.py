import pickle
import json
import hashlib
import os
import sys
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from attacks import dump_adv
import utils
from detection import uap_detector
from detection import blur_detector
from detection import bn_detector
import random
import inspect
from scipy.stats import beta
from scipy.stats import binom

# valid detector classes for ensembles
DET_CLASSES = ['UAPDetector', 'BlurDetector', 'BNDetector', 'LinearEnsembleDetector', 'LogRegEnsembleDetector', 'CalLogRegEnsembleDetector']

class Detector(object):

    def __init__(self, detector_dict, gift_basepath='/gift/', overwrite=False,
                 caldatasubpath='calibration/data', calmodelsubpath='calibration/fitted'):
        """
        Base class for trojan detection
        :param detector_dict: dictionary defining the detector operation
        :param gift_basepath: path to the gift root directory
        :param overwrite: if this is set, calibration should overwrite previous calibration files.
        """

        self.detector_dict = detector_dict
        self.gift_basepath = gift_basepath
        self.overwrite = overwrite
        self.imagenet_path = os.path.join(gift_basepath, 'data', 'random_imagenet_examples')

        self.caldatasubpath = caldatasubpath
        self.calmodelsubpath = calmodelsubpath

        # for accurate testing, this should be run before the top level process:
        # for filename in os.listdir(scratch_dirpath):
        #     filepath = os.path.join(scratch_dirpath, filename)
        #     os.remove(filepath)

    def cal(self, mod_rootdir=None, model_dirpaths=None, example_dirname='clean_example_data', n_samples=100):
        """
        Calibrates the detector on a set of models & returns the calibrated scores
        :param mod_rootdir: directory containing a bunch of model directories to be used for calibration. Either this or
         model_dirpaths should be set.
        :param model_dirpaths: list of model directories to be used for calibration. Either this or mod_rootdir should
        be set.
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: numpy array of calibrated probabilities, ordered like the model_dirpaths (or sorted directories in
        mod_rootdir)
        """
        # deal with a set of models & fit stuff
        # take either a folder and cal on all subfolders, or cal on modelpaths
        # run the detector on each model
        # calibrate
        # save irmodel & caldata
        raise NotImplementedError

    def detect_cal(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Run the detector on one model & return calibrated probability of trojan. Should match probability returned by
        "cal" method
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: probability of trojan
        """
        # deal with one model
        # do att (if necessary)
        # if att
        raise NotImplementedError

    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """
        raise NotImplementedError

    def get_cal_data(self, model_dirpaths, example_dirname, n_samples=100):
        """
        Runs the uncalibrated detector on the given models and returns the data ready for calibration.  Reads existing
        data if possible, unless overwrite flag is set
        :param model_dirpaths: list of model directories to compute calibration data for
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: mags (numpy array of raw score magnitudes)
        """
        raise NotImplementedError

    def compute_att(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Run the attack described by the detectors' attack dict (if any)
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: None
        """

        attpath = os.path.join(scratch_dirpath, self.get_attfn())
        if attpath is None or os.path.exists(attpath):
            return

        att_dict = self.detector_dict['att_dict']

        assert 'type' in att_dict, 'Need to specify an attack type'
        assert att_dict['type'] in ['l1', 'filt'], 'attack type must be l1 or filt'

        print('starting attack on ', model_filepath, "with attack", utils.get_hash(att_dict))

        if att_dict['type'] == 'l1':
            dump_adv.dump_model_l1(model_filepath, examples_dirpath, attpath,
                                   random_examples_dirpath=self.imagenet_path, **att_dict['kwargs'])
        else:
            dump_adv.dump_model_filt_attack(model_filepath, examples_dirpath, attpath,
                                            random_examples_dirpath=self.imagenet_path, **att_dict['kwargs'])

    def get_caldatapath(self, scratch_dirpath):
        """
        Gets the path for the saved calibration data
        :return: path for calibration data
        """
        fn = utils.get_hash(self.detector_dict) + '_cal.p'
        return os.path.join(scratch_dirpath, fn)

    def get_irpath(self):
        """
        Gets the path for the saved calibration joblib model
        :return: path for calibration model
        """
        fn = utils.get_hash(self.detector_dict) + '_ir.joblib'
        return os.path.join(self.gift_basepath, self.calmodelsubpath, fn)

    def get_attfn(self):
        """
        Gets the path for the detectors' attack numpy file
        :return: path for the numpy file or None if there is no attack dict
        """
        if 'att_dict' in self.detector_dict:
            return utils.get_hash(self.detector_dict['att_dict']) + '_att.npy'
        else:
            return None


class SoloDetector(Detector):
    """
    Abstract subclass of Detector for a single detector
    """
    # def get_cal_data(self, model_dirpaths, example_dirname):
    #     """
    #     Runs the uncalibrated detector on the given models and returns the data ready for calibration.  Reads existing
    #     data if possible, unless overwrite flag is set
    #     :param model_dirpaths: list of model directories to compute calibration data for
    #     :param example_dirname: name of the (clean) example data directory in each model directory
    #     :return: mags (numpy array of raw score magnitudes), y (the true classes)
    #     """
    #     caldatapath = self.get_caldatapath()
    #
    #     # check for existing data
    #     if os.path.exists(caldatapath) and not self.overwrite:
    #         print('loaded existing cal data')
    #         with open(caldatapath, 'rb')as f:
    #             model_dirpaths, mags, y = pickle.load(f)
    #     else:
    #         # compute & save new cal data
    #         print('no existing cal data found, starting a new run')
    #         mags = []
    #         y = []
    #         for model_dirpath in model_dirpaths:
    #             print('computing cal data for', model_dirpath)
    #             model_filepath = os.path.join(model_dirpath, 'model.pt')
    #             examples_dirpath = os.path.join(model_dirpath, example_dirname)
    #             scratch_dirpath = os.path.join(model_dirpath, 'scratch')
    #             truth_fn = os.path.join(model_dirpath, 'config.json')
    #
    #             if not os.path.exists(scratch_dirpath):
    #                 os.makedirs(scratch_dirpath)
    #
    #             mag = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)
    #             mags.append(mag)
    #             y.append(utils.get_class(truth_fn, classtype='binary', file=True))
    #
    #         mags = np.array(mags)
    #         y = np.array(y)
    #         with open(caldatapath, 'wb') as f:
    #             pickle.dump([model_dirpaths, mags, y], f)
    #     return mags, y

    def get_cal_data(self, model_dirpaths, example_dirname, n_samples=100):
        """
        Runs the uncalibrated detector on the given models and returns the data ready for calibration.  Reads existing
        data if possible, unless overwrite flag is set
        :param model_dirpaths: list of model directories to compute calibration data for
        :param example_dirname: name of the (clean) example data directory in each model directory
        :param n_samples: number of noisy samples of the data
        :return: mags... numpy array of raw score magnitudes with size: (len(model_dirpaths), n_samples)
        """
        mags = []
        for model_dirpath in model_dirpaths:
            scratch_dirpath = os.path.join(model_dirpath, 'scratch')
            if not os.path.exists(scratch_dirpath):
                os.makedirs(scratch_dirpath)
            caldatapath = self.get_caldatapath(scratch_dirpath)
            if os.path.exists(caldatapath):
                print('loading existing cal data from', caldatapath)
                with open(caldatapath, 'rb')as f:
                     mag = pickle.load(f)
            else:
                print('computing cal data for', model_dirpath)
                model_filepath = os.path.join(model_dirpath, 'model.pt')
                examples_dirpath = os.path.join(model_dirpath, example_dirname)

                mag = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)
                with open(caldatapath, 'wb') as f:
                    pickle.dump(mag, f)
            if n_samples is not None:
                noisy_mags = self.add_mag_noise(mag, n_samples)
                mags.append(noisy_mags)
            else:
                mags.append(np.array([mag]))
        mags = np.stack(mags)

        return mags

    def cal(self, mod_rootdir=None, model_dirpaths=None, example_dirname='clean_example_data', n_samples=100):
        """
        Implements calibration
        :param mod_rootdir: directory containing a bunch of model directories to be used for calibration. Either this or
         model_dirpaths should be set.
        :param model_dirpaths: list of model directories to be used for calibration. Either this or mod_rootdir should
        be set.
        :param example_dirname: name of the (clean) example data directory in each model directory
        :param n_samples: number of noisy samples of each data point
        :return: numpy array of calibrated probabilities, ordered like the model_dirpaths (or sorted directories in
        mod_rootdir)
        """

        assert (mod_rootdir is not None) != (model_dirpaths is not None), "set either mod_rootdir or model_dirpaths"
        if model_dirpaths is None:
            print("deprecation warning: using mod_rootdir is deprecated in favor of explicitly setting model_dirpaths")
            model_dirpaths = utils.get_modeldirs(mod_rootdir)

        # get the data for calibration
        mags = self.get_cal_data(model_dirpaths, example_dirname, n_samples=n_samples)
        mags = mags.reshape(-1)
        y = np.array([utils.get_class(os.path.join(pth, 'config.json'), classtype='binary', file=True) for pth in
                      model_dirpaths])
        if n_samples is not None:
            y = y.reshape(-1, 1) * np.ones([1, n_samples])
            y = y.reshape(-1)



        # check for saved model
        irpath = self.get_irpath()
        if os.path.exists(irpath) and not self.overwrite:
            ir_model = joblib.load(irpath)
        else:
            # run the calibration & save model
            ir_model = IsotonicRegression(out_of_bounds='clip')
            clippedmags = np.clip(mags, np.percentile(mags, 10), np.percentile(mags, 90))
            # clippedmags = np.clip(mags, np.percentile(mags, 25), np.percentile(mags, 75))

            ir_model.fit(clippedmags, y)
            joblib.dump(ir_model, irpath)

        # get & return the calibrated probabilities
        pcal = ir_model.transform(mags)
        return pcal

    def detect_cal(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Run the detector on one model & return calibrated probability of trojan. Should match probability returned by
        "cal" method
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: probability of trojan
        """

        mag = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)
        irpath = self.get_irpath()
        ir_model = joblib.load(irpath)
        pcal = ir_model.transform([mag])[0]
        print("solo detection, mag:",mag,"pcal",pcal)
        return pcal

    def add_mag_noise(self, mag, n_samples):
        """
        Fallback (noiseless) method for adding noise to magnitudes
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :param n_samples: number of noisy samples to return
        :return: noisy samples of probability of trojan
        """
        # mags = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)

        samples = np.ones(n_samples) * mag
        return samples


class UAPDetector(SoloDetector):
    """
    Implementation of SoloDetector for UAP detection
    """
    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """

        self.compute_att(model_filepath, examples_dirpath, scratch_dirpath)
        attpath = os.path.join(scratch_dirpath, self.get_attfn())

        assert 'type' in self.detector_dict, 'Need to specify an detector type'
        assert self.detector_dict['type'] in ['diff', 'filt'], 'detector type must be diff or filt'

        print("computing uap score on", model_filepath, 'with detector', utils.get_hash(self.detector_dict), 'and attack', utils.get_hash(self.detector_dict['att_dict']))
        if self.detector_dict['type'] == 'diff':
            mag = uap_detector.get_foolrate_diff(model_filepath, examples_dirpath, attpath, **self.detector_dict['kwargs'])
        else:
            filt_path = dump_adv.get_filt_path(attpath)
            mag = uap_detector.get_foolrate_filt(model_filepath, examples_dirpath, filt_path, **self.detector_dict['kwargs'])

        return mag


    def add_mag_noise(self, mag, n_samples):
        """
        Fallback (noiseless) method for adding noise to magnitudes
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :param n_samples: number of noisy samples to return
        :return: noisy samples of probability of trojan
        """
        # mags = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)

        # samples = np.ones(n_samples) * mag
        # return samples


        # mag = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)
        if self.detector_dict['type'] == 'diff':
            func = uap_detector.get_foolrate_diff
        else:
            func = uap_detector.get_foolrate_filt

        argspec = inspect.getfullargspec(func)
        keys = argspec.args[-len(argspec.defaults):]
        kwargs = {k:v for k,v in zip(keys, argspec.defaults)}
        for k, v in self.detector_dict['kwargs'].items():
            kwargs[k] = v
        nbatches = kwargs['nbatches']
        batchsz = kwargs['batchsz']

        ntrials = nbatches * batchsz
        a = 1 + mag*ntrials
        b = 1 + (1-mag)*ntrials
        samples = beta.rvs(a, b, size=n_samples)

        samples = binom.rvs(ntrials, samples, size=n_samples)/ntrials


        return samples


class BlurDetector(SoloDetector):
    """
    Implementation of SoloDetector for blur-based detection
    """
    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """

        self.compute_att(model_filepath, examples_dirpath, scratch_dirpath)
        attpath = os.path.join(scratch_dirpath, self.get_attfn())

        print("computing blur score on", model_filepath, 'with detector', utils.get_hash(self.detector_dict),
              'and attack', utils.get_hash(self.detector_dict['att_dict']))
        mag = blur_detector.get_blur_mag(attpath, sigma=2.0)

        return mag


class BNDetector(SoloDetector):
    """
    Implementation of Detector for blur-based detection
    """
    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """
        print("computing batch norm score on", model_filepath, 'with detector', utils.get_hash(self.detector_dict))
        mag = bn_detector.get_accdrop(model_filepath, examples_dirpath)
        return mag




class EnsembleDetector(Detector):
    """
    Abstract subclass of Detector for an ensemble of detectors
    """
    def __init__(self, detector_dict, **kwargs):
        super().__init__(detector_dict, **kwargs)

        # instantiate component detectors
        self.components = []
        assert 'components' in detector_dict, "EnsembleDetector must have a list in detector_dict['components']"

        for component_def in detector_dict['components']:

            # assert 'type' in component_def and 'detector_dict' in component_def, \
            #     "each component definition must have a 'type' and a 'detector_dict'"
            assert component_def['det_class'] in DET_CLASSES, "det_class must be in " + str(DET_CLASSES)

            thismodule = sys.modules[__name__]
            det_class = getattr(thismodule, component_def['det_class'])
            component = det_class(component_def, gift_basepath=self.gift_basepath, overwrite=self.overwrite,
                caldatasubpath=self.caldatasubpath)
            self.components.append(component)

    def cal(self, mod_rootdir=None, model_dirpaths=None, example_dirname='clean_example_data', n_samples=100):
        """
        Implements calibration
        :param mod_rootdir: directory containing a bunch of model directories to be used for calibration. Either this or
         model_dirpaths should be set.
        :param model_dirpaths: list of model directories to be used for calibration. Either this or mod_rootdir should
        be set.
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: numpy array of calibrated probabilities, ordered like the model_dirpaths (or sorted directories in
        mod_rootdir)
        """

        assert (mod_rootdir is not None) != (model_dirpaths is not None), "set either mod_rootdir or model_dirpaths"
        if model_dirpaths is None:
            print("deprecation warning: using mod_rootdir is deprecated in favor of explicitly setting model_dirpaths")
            model_dirpaths = utils.get_modeldirs(mod_rootdir)

        y = np.array([utils.get_class(os.path.join(pth, 'config.json'), classtype='binary', file=True) for pth in model_dirpaths])
        y = y.reshape(-1, 1) * np.ones([1, n_samples])
        y = y.reshape(-1)

        x = np.zeros([len(y), len(self.components)])
        order = [i for i in range(len(self.components))]
        random.shuffle(order)
        for i in order:
            print('starting calibration for ensemble component', i)
            component = self.components[i]
            pcal = component.cal(model_dirpaths=model_dirpaths, **kwargs)
            x[:, i] = pcal

        pcal = self.ens_cal(x, y)
        return pcal

    def ens_cal(self, x, y):
        """
        Calibrates the ensemble
        :param x: n-by-m array of n models with m component scores
        :param y: size-n array of classes for each model
        :return: numpy array of calibrated probabilities
        """
        raise NotImplementedError




class LogRegEnsembleDetector(EnsembleDetector):
    """
    Implementation of EnsembleDetector with a simpler logistic regression ensemble
    """
    def __init__(self, detector_dict, **kwargs):
        super().__init__(detector_dict, **kwargs)
        self.C = detector_dict["C"] if "C" in detector_dict else 100

    def get_wpath(self):
        """
        Gets the path to the saved linear model
        :return: path for the pickle file
        """
        fn = utils.get_hash(self.detector_dict) + '_lr.joblib'
        return os.path.join(self.gift_basepath, 'calibration', 'fitted', fn)

    def ens_cal(self, x, y):
        """
        Calibrates the ensemble
        :param x: n-by-m array of n models with m component scores
        :param y: size-n array of classes for each model
        :param C: regulariation term for LR. Use 10000.0 on 500+ models, 100.0 for ~100 models
        :return: numpy array of calibrated probabilities
        """
        wpath = self.get_wpath()

        # check for saved models
        if os.path.exists(wpath) and not self.overwrite:
            lr_model = joblib.load(wpath)
        else:
            lr_model = LogisticRegression(C=self.C, max_iter=10000, tol=1e-4)
            lr_model.fit(x, y)
            joblib.dump(lr_model, wpath)

        # compute trojan probabilities
        pcal = lr_model.predict_proba(x)[:, 1]

        # print a quick evaluation of the ensemble
        kld = log_loss(y, pcal)
        roc = roc_auc_score(y, pcal)
        print('ensemble kld:', kld, 'final auc', roc)

        return pcal

    def detect_cal(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Implements Detector.detect_cal
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: numpy array of calibrated probabilities
        """
        # read in models
        wpath = self.get_wpath()
        lr_model = joblib.load(wpath)

        # compute component probabilties
        x = self.detect_mag(model_filepath, examples_dirpath, scratch_dirpath)
        # x = [component.detect_mag(*args) for component in self.components]
        # x = np.array(x)

        # compute ensemble scores/probs
        pcal = lr_model.predict_proba([x])[0,1]

        # clip extremes to avoid risk of infinite CE
        pcal = np.clip(pcal, 0.025, 0.975)
        return pcal

    def cal(self, mod_rootdir=None, model_dirpaths=None, example_dirname='clean_example_data', n_samples=100):
        """
        Implements calibration
        :param mod_rootdir: directory containing a bunch of model directories to be used for calibration. Either this or
         model_dirpaths should be set.
        :param model_dirpaths: list of model directories to be used for calibration. Either this or mod_rootdir should
        be set.
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: numpy array of calibrated probabilities, ordered like the model_dirpaths (or sorted directories in
        mod_rootdir)
        """

        assert (mod_rootdir is not None) != (model_dirpaths is not None), "set either mod_rootdir or model_dirpaths"
        if model_dirpaths is None:
            print("deprecation warning: using mod_rootdir is deprecated in favor of explicitly setting model_dirpaths")
            model_dirpaths = utils.get_modeldirs(mod_rootdir)

        # y = [utils.get_class(os.path.join(pth, 'config.json'), classtype='binary', file=True) for pth in model_dirpaths]


        x = self.get_cal_data(model_dirpaths, example_dirname, n_samples=n_samples)

        # x = np.zeros([len(y), len(self.components)])
        # order = [i for i in range(len(self.components))]
        # random.shuffle(order)
        # for i in order:
        #     print('starting calibration for ensemble component', i)
        #     component = self.components[i]
        #     mags = component.get_cal_data(model_dirpaths, example_dirname)
        #     x[:, i] = mags
        y = np.array([utils.get_class(os.path.join(pth, 'config.json'), classtype='binary', file=True) for pth in
                      model_dirpaths])
        y = y.reshape(-1, 1) * np.ones([1, n_samples])
        y = y.reshape(-1)

        pcal = self.ens_cal(x, y)
        return pcal

    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """
        x = [component.detect_mag(model_filepath, examples_dirpath, scratch_dirpath) for component in self.components]
        x = np.array(x)
        return x

    def get_cal_data(self, model_dirpaths, example_dirname, n_samples=100):
        """
        Runs the uncalibrated detector on the given models and returns the data ready for calibration.  Reads existing
        data if possible, unless overwrite flag is set
        :param model_dirpaths: list of model directories to compute calibration data for
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: mags (numpy array of raw score magnitudes), y (the true classes)
        """
        if n_samples is None:
            x = np.zeros([len(model_dirpaths), len(self.components)])
        else:
            x = np.zeros([len(model_dirpaths)*n_samples, len(self.components)])
        order = [i for i in range(len(self.components))]
        random.shuffle(order)
        for i in order:
            print('starting calibration for ensemble component', i)
            component = self.components[i]
            mags = component.get_cal_data(model_dirpaths, example_dirname, n_samples=n_samples).reshape(-1)
            x[:, i] = mags


        # x = []
        # for component in self.components:
        #     mags = component.get_cal_data(*args)
        #     x.append(mags)
        # x = np.array(x).transpose()

        return x


class CalLogRegEnsembleDetector(LogRegEnsembleDetector):
    def detect_mag(self, model_filepath, examples_dirpath, scratch_dirpath):
        """
        Compute the raw (uncalibrated) detection score for a model
        :param model_filepath: standard input argument
        :param examples_dirpath: standard input argument
        :param scratch_dirpath: standard input argument
        :return: detection score
        """
        x = [component.detect_cal(model_filepath, examples_dirpath, scratch_dirpath) for component in self.components]
        x = np.array(x)
        return x

    def get_cal_data(self, model_dirpaths, example_dirname, n_samples=100):
        """
        Runs the uncalibrated detector on the given models and returns the data ready for calibration.  Reads existing
        data if possible, unless overwrite flag is set
        :param model_dirpaths: list of model directories to compute calibration data for
        :param example_dirname: name of the (clean) example data directory in each model directory
        :return: mags (numpy array of raw score magnitudes), y (the true classes)
        """
        if n_samples is None:
            x = np.zeros([len(model_dirpaths), len(self.components)])
        else:
            x = np.zeros([len(model_dirpaths)*n_samples, len(self.components)])
        order = [i for i in range(len(self.components))]
        random.shuffle(order)
        for i in order:
            print('starting calibration for ensemble component', i)
            component = self.components[i]
            mags = component.cal(model_dirpaths=model_dirpaths, example_dirname=example_dirname, n_samples=n_samples)
            x[:, i] = mags

        return x
