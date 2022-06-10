import os
import numpy as np
import utils.r7utils as r7utils
import pickle
import utils
from detection.oldr7functions import get_th_feats_batch3
import random
import trinv


def demo(seed=1, ignore_grads=False, loss_cutoff=-np.inf, test_model = 'id-00000040'):
    base_path = 'data/round7/models'
    # test_model = 'id-00000040'
    example_folder_name = 'clean_example_data'
    model_dirpath = os.path.join(base_path, test_model)
    examples_dirpath = os.path.join(model_dirpath, example_folder_name)
    model_filepath = os.path.join(model_dirpath, 'model.pt')

    class struct():
        def __init__(self):
            self.trigger_token_length = 5
            # self.trigger_token_length = 2
            self.topk_candidate_tokens = 20
            self.total_num_update = 1
            self.combine_class_pairs = False
            self.n_repeats = 1

    args = struct()


    trinv.run_trigger_search_on_model(args, model_filepath, examples_dirpath, seed_num=seed, ignore_grads=ignore_grads, loss_cutoff=loss_cutoff)


def demo_repeat(seed=1, ignore_grads=False):
    base_path = 'data/round7/models'
    example_folder_name = 'clean_example_data'
    test_models = ['id-00000040', 'id-00000077', 'id-00000084', 'id-00000085', 'id-00000086', 'id-00000099', 'id-00000103', 'id-00000125', 'id-00000145', 'id-00000154', 'id-00000166']
    for test_model in test_models:
        print('starting', test_model)
        model_dirpath = os.path.join(base_path, test_model)
        examples_dirpath = os.path.join(model_dirpath, example_folder_name)
        model_filepath = os.path.join(model_dirpath, 'model.pt')

        class struct():
            def __init__(self):
                self.trigger_token_length = 3
                # self.trigger_token_length = 2
                self.topk_candidate_tokens = 100
                self.total_num_update = 2
                self.combine_class_pairs = False
                self.n_repeats = 5

        args = struct()


        trinv.run_trigger_search_on_model(args, model_filepath, examples_dirpath, seed_num=seed, ignore_grads=ignore_grads)

def cheat_sc(seed=1, trigger_token_length=3, topk_candidate_tokens=100, total_num_update=2, n_repeats=5, randloc=True):
    base_path = 'data/round7/models'
    example_folder_name = 'clean_example_data'

    srctgts = {'id-00000040':(5,3), 'id-00000077':(9,11), 'id-00000084':(1,5), 'id-00000085':(3,1), 'id-00000086':(3,7), 'id-00000099':(5,1),
                   'id-00000103':(5,1), 'id-00000125': (3,5), 'id-00000145': (3,7), 'id-00000154': (11,1), 'id-00000166': (9,5)}

    srctgts2 = {'id-00000012':(3,5), 'id-00000013':(5,11), 'id-00000033':(5,3), 'id-00000039':(7,5), 'id-00000070':(1,3), 'id-00000096':(1,5),
                   'id-00000104':(1,11), 'id-00000107':(7,3), 'id-00000123':(1,5), 'id-00000158':(5,1), 'id-00000163':(1,7), 'id-00000164':(7,1)}
    srctgts = {**srctgts, **srctgts2}

    srctgts2 = {'id-00000030':(3,5), 'id-00000102':(7,5), 'id-00000162':(7,3)}
    srctgts = {**srctgts, **srctgts2}


    if randloc:
        # global phrase triggers
        test_models = ['id-00000040', 'id-00000077', 'id-00000084', 'id-00000085', 'id-00000086', 'id-00000099',
                       'id-00000103', 'id-00000125', 'id-00000145', 'id-00000154', 'id-00000166']
    else:
        # local phrase triggers
        test_models = ['id-00000012', 'id-00000013', 'id-00000033', 'id-00000039', 'id-00000070', 'id-00000096',
                        'id-00000104', 'id-00000107', 'id-00000123', 'id-00000158', 'id-00000163', 'id-00000164']

    kwargs = {'randloc': randloc,
              'trigger_token_length': trigger_token_length,
              'total_num_update': total_num_update,
              'n_repeats': n_repeats,
              'topk_candidate_tokens': topk_candidate_tokens}

    for test_model in test_models:
        print('starting', test_model)
        srctgt = srctgts[test_model]
        model_dirpath = os.path.join(base_path, test_model)
        examples_dirpath = os.path.join(model_dirpath, example_folder_name)
        model_filepath = os.path.join(model_dirpath, 'model.pt')

        trinv.run_trigger_search_on_model(model_filepath, examples_dirpath, seed_num=seed, srctgtlist=[srctgt], **kwargs)

from sklearn.metrics import roc_auc_score


def auc_sc(seed=1, trigger_token_length=3, topk_candidate_tokens=100, total_num_update=2, n_repeats=5, randloc=True):
    base_path = 'data/round7/models'
    example_folder_name = 'clean_example_data'

    if randloc:
        # global phrase triggers
        modellist = ['id-00000040', 'id-00000077', 'id-00000084', 'id-00000085', 'id-00000086', 'id-00000099',
                       'id-00000103', 'id-00000125', 'id-00000145', 'id-00000154', 'id-00000166']
    else:
        # local phrase triggers
        modellist = ['id-00000012', 'id-00000013', 'id-00000033', 'id-00000039', 'id-00000070', 'id-00000096',
                        'id-00000104', 'id-00000107', 'id-00000123', 'id-00000158', 'id-00000163', 'id-00000164']
    # add in clean models
    modellist += ['id-00000030', 'id-00000102', 'id-00000162', 'id-00000001', 'id-00000002', 'id-00000004', 'id-00000008', 'id-00000009', 'id-00000010', 'id-00000011', 'id-00000014', 'id-00000017']

    kwargs = {'randloc': randloc,
              'trigger_token_length': trigger_token_length,
              'total_num_update': total_num_update,
              'n_repeats': n_repeats,
              'topk_candidate_tokens': topk_candidate_tokens}

    scores = []
    classes = []

    for test_model in modellist:
        print('starting', test_model)
        model_dirpath = os.path.join(base_path, test_model)
        examples_dirpath = os.path.join(model_dirpath, example_folder_name)
        model_filepath = os.path.join(model_dirpath, 'model.pt')
        config_filepath = os.path.join(model_dirpath, 'config.json')

        cls = utils.get_class(config_filepath)
        if cls==1:
            truth = utils.read_truthfile(config_filepath)
            if truth['triggers'][0]["trigger_executor_name"] == 'character':
                continue

        loss_return, _, _ = trinv.run_trigger_search_on_model(model_filepath, examples_dirpath, seed_num=seed, srctgtlist=None, **kwargs)

        scores.append(loss_return)
        classes.append(cls)

    print(roc_auc_score(classes,scores))

def auc2x_sc(seed=1):
    base_path = 'data/round7/models'
    example_folder_name = 'clean_example_data'
    output_path = './r7full'
    os.makedirs(output_path, exist_ok=True)

    modellist = os.listdir(base_path)
    random.shuffle(modellist)

    glo_kwargs = {'randloc': True,
                  'trigger_token_length': 3,
                  'total_num_update': 2,
                  'n_repeats': 5,
                  'topk_candidate_tokens': 100}

    loc_kwargs = {'randloc': False,
                  'trigger_token_length': 3,
                  'total_num_update': 2,
                  'n_repeats': 1,
                  'topk_candidate_tokens': 250}

    for test_model in modellist:
        print('starting', test_model)
        model_dirpath = os.path.join(base_path, test_model)
        examples_dirpath = os.path.join(model_dirpath, example_folder_name)
        model_filepath = os.path.join(model_dirpath, 'model.pt')
        config_filepath = os.path.join(model_dirpath, 'config.json')

        output_fn = os.path.join(output_path,test_model+'.p')
        if os.path.exists(output_fn):
            continue

        cls = utils.get_class(config_filepath)
        if cls==1:
            truth = utils.read_truthfile(config_filepath)
            if truth['triggers'][0]["trigger_executor_name"] == 'character':
                continue

        trigger_path_name = 'r7scratch/chr_triggerdir'
        trigger_dirpath = os.path.join('./', trigger_path_name)
        pfeats = get_th_feats_batch3(model_filepath, examples_dirpath, trigger_dirpath, tokenizer_filepath=None, nonzero_only=True)
        clean_preds = pfeats["clean_output"]
        triggered_preds = {k: v['trig_preds'] for k, v in pfeats["res"].items()}
        trigger_preds = {k: v['actual_trig_preds'] for k, v in pfeats["res"].items()}
        lcm = r7utils.localcharmetric(clean_preds, triggered_preds, trigger_preds=trigger_preds)

        glo_res = trinv.run_trigger_search_on_model(model_filepath, examples_dirpath, seed_num=seed, **glo_kwargs)
        loc_res = trinv.run_trigger_search_on_model(model_filepath, examples_dirpath, seed_num=seed, **loc_kwargs)

        with open(output_fn,'wb') as f:
            pickle.dump([glo_res, loc_res, lcm], f)


