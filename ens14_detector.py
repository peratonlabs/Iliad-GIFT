
import warnings
import detection.defs
import detection

warnings.filterwarnings("ignore")
ens_prefix = 'ens14_'

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='GIFT detector.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        default='./model.pt')
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        default='./output')
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        default='./scratch')
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                        default='./example')
    parser.add_argument('--gift_basepath', type=str,
                        help='File path to the folder where our trained detection or calibration models are.',
                        default='/gift/')

    args = parser.parse_args()

    ens_def = detection.defs.ens14
    det_class = getattr(detection, ens_def['det_class'])
    mydet = det_class(ens_def, gift_basepath=args.gift_basepath)

    trojan_probability = mydet.detect_cal(args.model_filepath, args.examples_dirpath, args.scratch_dirpath)

    print('Final Trojan Probability: {}'.format(trojan_probability))

    with open(args.result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

