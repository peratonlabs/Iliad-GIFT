import os
import random
import utils


def makesplit(mod_rootdir, exp_name, seed, train_ratio, test_ratio=None):

    random.seed(seed)
    model_dirpaths = utils.get_modeldirs(mod_rootdir)
    random.shuffle(model_dirpaths)

    train_split = round(train_ratio * len(model_dirpaths))

    train_models = model_dirpaths[:train_split]

    if test_ratio is None:
        test_models = model_dirpaths[train_split:]
    else:
        test_split = round(test_ratio * len(model_dirpaths)) + train_split
        test_models = model_dirpaths[train_split:test_split]

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


def make_stdexps():
    mod_rootdir = 'data/round3models'

    exp_base = 'r3_9010'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.9)

    exp_base = 'r3_8020'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.8)

    exp_base = 'r3_1010'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.1, 0.1)

    exp_base = 'r3_2525'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.25, 0.25)

    exp_base = 'r3_5050'
    for i in range(10):
        makesplit(mod_rootdir, exp_base + "_seed" + str(i), i, 0.5)

    makesplit(mod_rootdir, 'r3_all', 0, 1.0)


