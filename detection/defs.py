

################################# ATTACKS #######################################################

# att dict for untargeted L1 attack for round 3
r3_ut_l1_att = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 100, 'targeted': False}}
# att dict for random targeted L1 attack for round 3
r3_rt_l1_att = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 100, 'targeted': True, 'random_targets': True}}
# att dict for guessed targeted L1 attack for round 3
r3_gt_l1_att = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 100, 'targeted': True, 'random_targets': False}}
# att dict for untargeted filter attack for round 3
r3_ut_filt_att = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 100, 'targeted': False}}
# att dict for random targeted filter attack for round 3
r3_rt_filt_att = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 100, 'targeted': True, 'random_targets': True}}
# att dict for guessed targeted filter attack for round 3
r3_gt_filt_att = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 100, 'targeted': True, 'random_targets': False}}


# att dict for untargeted L1 attack for round 3, 300 images
r3_ut_l1_att_x3 = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 300, 'targeted': False}}
# att dict for random targeted L1 attack for round 3
r3_rt_l1_att_x3 = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 300, 'targeted': True, 'random_targets': True}}
# att dict for guessed targeted L1 attack for round 3
r3_gt_l1_att_x3 = {'type': 'l1', 'kwargs': {'eps': 2000., 'l1_sparsity': 0.99, 'nims': 300, 'targeted': True, 'random_targets': False}}
# att dict for untargeted filter attack for round 3
r3_ut_filt_att_x3 = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 300, 'targeted': False}}
# att dict for random targeted filter attack for round 3
r3_rt_filt_att_x3 = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 300, 'targeted': True, 'random_targets': True}}
# att dict for guessed targeted filter attack for round 3
r3_gt_filt_att_x3 = {'type': 'filt', 'kwargs': {'eps': 0.03, 'nims': 300, 'targeted': True, 'random_targets': False}}

################################# DETECTORS #######################################################


# untargeted L1 detectors
r3_ut_l1_edet = {'att_dict': r3_ut_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_l1_tdet = {'att_dict': r3_ut_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# random targeted L1 detectors
r3_rt_l1_edet = {'att_dict': r3_rt_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_rt_l1_tdet = {'att_dict': r3_rt_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# guessed targeted L1 detectors
r3_gt_l1_edet = {'att_dict': r3_gt_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_gt_l1_tdet = {'att_dict': r3_gt_l1_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}

# untargeted filter-difference detectors
r3_ut_filtdiff_edet = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_filtdiff_tdet = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# random targeted filter-difference detectors
r3_rt_filtdiff_edet = {'att_dict': r3_rt_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_rt_filtdiff_tdet = {'att_dict': r3_rt_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# guessed targeted filter-difference detectors
r3_gt_filtdiff_edet = {'att_dict': r3_gt_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_gt_filtdiff_tdet = {'att_dict': r3_gt_filt_att, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}

# untargeted filter detectors
r3_ut_filt_edet = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_tdet = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# random targeted filter detectors
r3_rt_filt_edet = {'att_dict': r3_rt_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_rt_filt_tdet = {'att_dict': r3_rt_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
# guessed targeted filter detectors
r3_gt_filt_edet = {'att_dict': r3_gt_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_gt_filt_tdet = {'att_dict': r3_gt_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}


# untargeted L1 blur detector
r3_ut_blur = {'att_dict': r3_ut_l1_att, 'det_class': 'BlurDetector', 'kwargs': {}}
# random targeted L1 blur detector
r3_rt_blur = {'att_dict': r3_rt_l1_att, 'det_class': 'BlurDetector', 'kwargs': {}}
# guessed targeted L1 blur detector
r3_gt_blur = {'att_dict': r3_gt_l1_att, 'det_class': 'BlurDetector', 'kwargs': {}}

# batch norm detector
r3_bn = {'det_class': 'BNDetector', 'kwargs': {}}

r3_ut_filtdiff_tdet_x3 = {'att_dict': r3_ut_filt_att_x3, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 30, 'training': True}}
r3_gt_l1_tdet_x3 = {'att_dict': r3_gt_l1_att_x3, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 30, 'training': True}}
r3_ut_blur_x3 = {'att_dict': r3_ut_l1_att_x3, 'det_class': 'BlurDetector', 'kwargs': {}}

r3_ut_l1_tdet_x3 = {'att_dict': r3_ut_l1_att_x3, 'det_class': 'UAPDetector', 'type': 'diff', 'kwargs': {'pert_scale': 1.0, 'nbatches': 30, 'training': True}}
r3_rt_filt_tdet_x3 = {'att_dict': r3_rt_filt_att_x3, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 30, 'training': True}}


r3_ut_filt_tdet_sc2 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_sc5 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_sc05 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_sc025 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': True}}


r3_ut_filt_att_01 = {'type': 'filt', 'kwargs': {'eps': 0.01, 'nims': 100, 'targeted': False}}
r3_ut_filt_att_003 = {'type': 'filt', 'kwargs': {'eps': 0.003, 'nims': 100, 'targeted': False}}
r3_ut_filt_att_1 = {'type': 'filt', 'kwargs': {'eps': 0.1, 'nims': 100, 'targeted': False}}

r3_ut_filt_tdet_att_01 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_003 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_1 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': True}}



r3_ut_filt_tdet_att_01_sc2 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_01_sc5 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_01_sc05 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_01_sc025 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': True}}

r3_ut_filt_tdet_att_003_sc2 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_003_sc5 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_003_sc05 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_003_sc025 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': True}}


r3_ut_filt_tdet_att_1_sc2 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_1_sc5 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_1_sc05 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': True}}
r3_ut_filt_tdet_att_1_sc025 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': True}}



r3_ut_filt_edet_att_01_sc2 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_01_sc5 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_01_sc05 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_01_sc025 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': False}}

r3_ut_filt_edet_att_003_sc2 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_003_sc5 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_003_sc05 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_003_sc025 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': False}}

r3_ut_filt_edet_att_1_sc2 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_1_sc5 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_1_sc05 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_1_sc025 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': False}}


r3_ut_filt_edet_sc2 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 2.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_sc5 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 5.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_sc05 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.5, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_sc025 = {'att_dict': r3_ut_filt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 0.25, 'nbatches': 10, 'training': False}}

r3_ut_filt_edet_att_01 = {'att_dict': r3_ut_filt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_003 = {'att_dict': r3_ut_filt_att_003, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_filt_edet_att_1 = {'att_dict': r3_ut_filt_att_1, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}


import numpy as np
r3_ut_inffilt_att = {'type': 'filt', 'kwargs': {'eps': 0.0003, 'nims': 100, 'targeted': False, 'ord': np.inf}}
r3_ut_inffilt_att_01 = {'type': 'filt', 'kwargs': {'eps': 0.001, 'nims': 100, 'targeted': False, 'ord': np.inf}}


r3_ut_inffilt_edet_att = {'att_dict': r3_ut_inffilt_att, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}
r3_ut_inffilt_edet_att_01 = {'att_dict': r3_ut_inffilt_att_01, 'det_class': 'UAPDetector', 'type': 'filt', 'kwargs': {'pert_scale': 1.0, 'nbatches': 10, 'training': False}}

################################# ENSEMBLES #######################################################

ens14 = {
    'det_class': 'LogRegEnsembleDetector',
    'components': [
                    r3_ut_l1_tdet,
                    r3_rt_l1_tdet,
                    r3_ut_inffilt_edet_att_01,
                    r3_ut_filt_tdet,
                    r3_ut_filt_tdet_att_1,
                    r3_ut_filt_tdet_att_01_sc2,
                    r3_ut_filt_edet_att_01_sc2,
                    r3_ut_filt_tdet_att_003_sc5
    ]}

