import sys
sys.path.append(".")
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from joblib import dump, load
import torch
import skimage.io
import pickle
# from advertorch.attacks import PGDAttack
from attacks.attack import PGDAttack, PGDAttackOrig
import random
# from utils import Filt_Model
import time
# from detection import uap_detector
from utils import get_target_class, Filt_Model, readimages
import utils

# EXAMPLES_FOLDER_NAME = 'example_data' # round2
# EXAMPLES_FOLDER_NAME = 'clean_example_data' # round3
# BASE_FOLDER = 'data/round2models' # round2
# BASE_FOLDER = 'data/round3models' # round3

# random_examples_dirpath = "./data/random_imagenet_examples"

def get_filt_path(outpath):
    if outpath[-4:] =='.npy':
        outpath = outpath[:-4]
    return outpath + '_filters' + '.npy'


def dump_model_filt_attack(model_filepath, examples_dirpath, outpath, eps=0.003, nb_iter=10, batchsz=20, nims=20,
                           targeted=False, random_targets=True, savediffs=True, ord=2,
                           random_examples_dirpath="./data/random_imagenet_examples"):
    """
    
    :param model_path: (string) path to the model.pt file in question
    :param example_dir: (string) path to the example_data directory
    :param outpath: 
    :param eps: 
    :param nb_iter: (int) number of batches.  total samples used for computing foolrate is nbatches*batchsz
    :param batchsz: (int) batch size for evaluation, lowering this reduces memory consumption
    :param nims: 
    :param targeted:
    :param savediffs:
    :param ord:
    :return: 
    """
    # print('starting ', model_filepath)
    # print('outputting to  ', outpath)
    eps = float(eps)
    model = torch.load(model_filepath).cuda()
    model.eval()

    filt_sz = 15
    mod_model = Filt_Model(model, None, filt_sz=filt_sz, add_bias=True)

    att = PGDAttack(mod_model, loss_fn=None, eps=eps, nb_iter=nb_iter,
                    eps_iter=eps/3, rand_init=True,
                    ord=ord, targeted=targeted)

    
    ims = utils.readimages(examples_dirpath, nims=nims).cuda()
    ims_ = []
    filters = []
    # tgts = []

    cur_ims = ims[0:1]
    ncls = model(cur_ims).shape[1]

    nchan = 3
    zero_filt = torch.zeros([nchan * batchsz, nchan+1, filt_sz, filt_sz]).cuda()
    for i in range(int(np.ceil(ims.shape[0] / batchsz))):

        cur_ims = ims[i * batchsz:(i + 1) * batchsz]
        if cur_ims.shape[0]<batchsz:
            zero_filt = torch.zeros([nchan * cur_ims.shape[0], nchan + 1, filt_sz, filt_sz]).cuda()
        mod_model.set_images(cur_ims)

        if targeted:
            if random_targets:
                tgt = torch.randint(ncls, [cur_ims.shape[0]]).cuda()
            else:
                tgt = torch.randint(ncls, [cur_ims.shape[0]]).cuda()*0
                tgt += get_target_class.guess_target_class(model_filepath, random_examples_dirpath, num_sample=100)
            # tgts.append(tgt)
            att_filt = att.perturb(zero_filt, y=tgt)
        else:
            att_filt = att.perturb(zero_filt)
        cur_pert_ims = mod_model.filter(att_filt)
        filters.append(att_filt.cpu().detach().numpy())
        ims_.append(cur_pert_ims)

    # if targeted:
    #     tgts = torch.cat(tgts).cpu().detach().numpy()
    ims_ = torch.cat(ims_)
    ims_ = ims_.cpu().detach().numpy()
    ims = ims.cpu().detach().numpy()
    if savediffs:
        np.save(outpath, np.stack([ims, ims_]))


    np.save(get_filt_path(outpath), filters)

    # if outpath[-4:] =='.npy':
    #     outpath = outpath[:-4]
    # np.save(outpath + '_filters', filters)


# def dump_model_filt_attack(model_filepath, examples_dirpath, outpath, eps=0.003, nb_iter=10, batchsz=20, nims=20):
#     # print('starting ', model_filepath)
#     # print('outputting to  ', outpath)
#     eps = float(eps)
#     model = torch.load(model_filepath).cuda()
#     model.eval()
#
#     filt_sz = 15
#     mod_model = Filt_Model(model, None, filt_sz=filt_sz, add_bias=True)
#
#     att = PGDAttack(mod_model, loss_fn=None, eps=eps, nb_iter=nb_iter,
#                     eps_iter=eps/3, rand_init=True,
#                     ord=2, targeted=False)
#
#     import uap_detector
#     ims = uap_detector.readimages(examples_dirpath, nims=nims).cuda()
#     ims_ = []
#     filters = []
#
#     nchan = 3
#     zero_filt = torch.zeros([nchan * batchsz, nchan+1, filt_sz, filt_sz]).cuda()
#     for i in range(int(np.ceil(ims.shape[0] / batchsz))):
#
#         cur_ims = ims[i * batchsz:(i + 1) * batchsz]
#         if cur_ims.shape[0]<batchsz:
#             zero_filt = torch.zeros([nchan * cur_ims.shape[0], nchan + 1, filt_sz, filt_sz]).cuda()
#         mod_model.set_images(cur_ims)
#         att_filt = att.perturb(zero_filt)
#
#         cur_pert_ims = mod_model.filter(att_filt)
#
#         filters.append(att_filt.cpu().detach().numpy())
#
#
#         ims_.append(cur_pert_ims)
#     ims_ = torch.cat(ims_)
#     ims_ = ims_.cpu().detach().numpy()
#     ims = ims.cpu().detach().numpy()
#     np.save(outpath, np.stack([ims, ims_]))
#     if outpath[-4:] =='.npy':
#         outpath = outpath[:-4]
#     np.save(outpath + '_filters', filters)


def dump_model_l1(model_filepath, examples_dirpath, outpath, eps=100.0, nb_iter=10, batchsz=20, nims=20,
                  l1_sparsity=0.99, targeted=False, random_targets=True, truetarg=False,
                  random_examples_dirpath="./data/random_imagenet_examples"):
    """
    
    :param model_filepath: (string) path to the model.pt file in question
    :param example_dir: (string) path to the example_data directory
    :param outpath: 
    :param eps: 
    :param nb_iter: (int) number of batches.  total samples used for computing foolrate is nbatches*batchsz
    :param batchsz: (int) batch size for evaluation, lowering this reduces memory consumption
    :param nims: 
    :param l1_sparsity:
    :param targeted:
    :return: 
    """    
    # print('starting ', model_filepath)
    eps = float(eps)
    model = torch.load(model_filepath).cuda()
    # model = torch.load(os.path.join(path, 'model.pt')).cuda()
    model.eval()

    # att = PGDAttack(model, loss_fn=None, eps=eps, nb_iter=nb_iter,
    #                 eps_iter=eps/3., rand_init=True, clip_min=0., clip_max=1.,
    #                 ord=1, targeted=False)
    att = PGDAttackOrig(model, loss_fn=None, eps=eps, nb_iter=nb_iter,
                    eps_iter=eps / 3., rand_init=True, clip_min=0., clip_max=1.,
                    ord=1, targeted=targeted, l1_sparsity=l1_sparsity)

    
    ims = readimages(examples_dirpath, nims=nims).cuda()

    cur_ims = ims[0:1]
    ncls = model(cur_ims).shape[1]
    # imfns = os.listdir(path=examples_dirpath)
    # nonims = []
    # for imfn in imfns:
    #     if imfn[-4:] != '.png':
    #         nonims.append(imfn)
    # for nonim in nonims:
    #     imfns.remove(nonim)
    #
    # random.shuffle(imfns)
    # imfns = imfns[:nims]


    # imfns = os.listdir(path=examples_dirpath)
    # print(examples_dirpath)
    # print(imfns)
    # ims = []
    # for imfn in imfns:
    #     im = readim(os.path.join(examples_dirpath, imfn))
    #     ims.append(im)

    # ims = torch.cat(ims).cuda()
    ims_ = []
    for i in range(int(np.ceil(ims.shape[0] / batchsz))):
        cur_ims = ims[i * batchsz:(i + 1) * batchsz]
        if targeted:
            if random_targets:
                tgt = torch.randint(ncls, [cur_ims.shape[0]]).cuda()
            else:
                tgt = torch.randint(ncls, [cur_ims.shape[0]]).cuda()*0

                if truetarg:
                    truth_fn = os.path.join(os.path.split(model_filepath)[0],'config.json')
                    tgt += utils.get_tgtclass(truth_fn)
                else:
                    tgt += get_target_class.guess_target_class(model_filepath, random_examples_dirpath, num_sample=100)


            # tgts.append(tgt)
            # att_filt = att.perturb(zero_filt, y=tgt)
            pert_im = att.perturb(cur_ims, y=tgt)
        else:
            pert_im = att.perturb(cur_ims)

        ims_.append(pert_im)
        # ims_.append(att.perturb(ims[i * batchsz:(i + 1) * batchsz]))
    ims_ = torch.cat(ims_)

    ims_ = ims_.cpu().detach().numpy()
    ims = ims.cpu().detach().numpy()

    # outfn = infn + suffix + '.npy'
    # if imfn[-6:] == '_0.png' and not os.path.exists(outfn):
    #     im = im2tensor(os.path.join(imdir, imfn)).cuda()
    #     im = im.cuda()
    #     im_ = att.perturb(im)
    #     im_ = im_.cpu().detach().numpy()

    np.save(outpath, np.stack([ims, ims_]))


def dump_adv(name, folder, example_folder_name='example_data', type='l1', nModels=None, **kwargs):
    """
    :param name:
    :param folder:
    :param type:
    """    
    dirs = os.listdir(path=folder)
    if nModels is not None:
        dirs.sort()
        dirs = dirs[:nModels]
    else:
        random.shuffle(dirs)

    print('starting ', len(dirs), 'files')
    i = 0
    # batchsz = 20

    for dir in dirs:

        path = os.path.join(folder, dir)
        # imdir = os.path.join(path, 'example_data')
        imdir = os.path.join(path, example_folder_name)

        outfn = os.path.join(imdir, name)

        model_filepath = os.path.join(path, 'model.pt')

        if os.path.exists(model_filepath) and not (os.path.exists(outfn + '.npy') or os.path.exists(outfn)):

            t=time.time()

            # print('starting ', dir)
            if type=='l1':
                dump_model_l1(model_filepath, imdir, outfn, **kwargs)
            elif type=='filt':
                # if targeted:
                #     dump_model_targfilt_attack(model_filepath, imdir, outfn, **kwargs)
                # else:
                dump_model_filt_attack(model_filepath, imdir, outfn, **kwargs)
            else:
                assert False, "pert type not supported"
            print('finished ', dir, 'time:',time.time()-t)




