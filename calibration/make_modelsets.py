import os
import random
import utils


def makesplit(mod_rootdir, exp_name, seed, train_ratio, test_ratio=None, testmod_rootdir=None):

    random.seed(seed)
    model_dirpaths = utils.get_modeldirs(mod_rootdir)
    random.shuffle(model_dirpaths)

    train_split = round(train_ratio * len(model_dirpaths))

    train_models = model_dirpaths[:train_split]

    if testmod_rootdir is None:
        if test_ratio is None:
            test_models = model_dirpaths[train_split:]
        else:
            test_split = round(test_ratio * len(model_dirpaths)) + train_split
            test_models = model_dirpaths[train_split:test_split]
    else:
        test_models = utils.get_modeldirs(testmod_rootdir)
        if test_ratio is not None:
            random.shuffle(test_models)
            test_split = round(test_ratio * len(test_models))
            test_models = model_dirpaths[:test_split]

    train_models = [fn + '\n' for fn in train_models]
    test_models = [fn + '\n' for fn in test_models]

    train_models.sort()
    test_models.sort()

    train_fn = os.path.join('calibration/modelsets', exp_name + '_trainset.txt')
    test_fn = os.path.join('calibration/modelsets', exp_name + '_testset.txt')

    with open(train_fn, "w") as f:
        f.writelines(train_models)
    with open(test_fn, "w") as f:
        f.writelines(test_models)





def make_stdexps_rnd5():
    mod_rootdir = 'data/round5models'

    exp_base = 'r5_9010'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.9)

    exp_base = 'r5_8020'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.8)

    exp_base = 'r5_1010'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.1, 0.1)

    exp_base = 'r5_2525'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.25, 0.25)

    exp_base = 'r5_5050'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.5)

    makesplit(mod_rootdir, 'r5_all', 0, 1.0)

