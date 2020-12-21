
import detection.defs
import argparse
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GIFT detector calibration routine.')
    parser.add_argument('--ens_name', type=str, default='ens14')
    # parser.add_argument('--mod_rootdir', type=str, default='data/round3models')
    parser.add_argument('--modellist', type=str, default='calibration/modelsets/r3_all_trainset.txt')
    parser.add_argument("--overwrite", action="store_true")
    # parser.add_argument("--shufflecal", action="store_true")
    args = parser.parse_args()

    import os
    if not os.path.exists("./calibration/fitted"):
        os.makedirs("./calibration/fitted")

    # get the detector definition
    ens_def = getattr(detection.defs, args.ens_name)

    # get the class from the definition
    det_class = getattr(detection, ens_def['det_class'])

    # instantiate the detector
    mydet = det_class(ens_def, gift_basepath='./', overwrite=args.overwrite)

    model_dirpaths = utils.get_modeldirs(args.modellist, usefile=True)
    mydet.cal(model_dirpaths=model_dirpaths)

