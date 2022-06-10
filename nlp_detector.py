
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from detection.sc_detection import det as sc_det
from detection.ner_detection import det as ner_det
from detection.qa_detection import det as qa_det

from detection.sc_detection import cal as sc_cal
from detection.ner_detection import cal as ner_cal
from detection.qa_detection import cal as qa_cal



import numpy as np
import utils



def configure(arg_dict):
    """
    :param arg_dict:
    
    """
    metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])
    print(f"Configure Mode Activated: Calibrating Detector(s)!")

    if arg_dict['task'] == 'qa':
        qa_cal(arg_dict, metaParameters["qa_dets"], metaParameters["qa_cls"])
    elif arg_dict['task'] == 'sc':
        sc_cal(arg_dict, metaParameters["sc_dets"], metaParameters["sc_cls"])
    elif arg_dict['task'] == 'ner':
        ner_cal(arg_dict, metaParameters["ner_dets"], metaParameters["ner_cls"])
    elif arg_dict['task'] == 'all':
        qa_cal(arg_dict, metaParameters["qa_dets"], metaParameters["qa_cls"])
        sc_cal(arg_dict, metaParameters["sc_dets"], metaParameters["sc_cls"])
        ner_cal(arg_dict, metaParameters["ner_dets"], metaParameters["ner_cls"])
    else:
        assert False, 'task for config needs to be one of [all, ner, sc, qa]'





if __name__ == "__main__":
    import json
    import jsonschema
    from jsonargparse import ArgumentParser, ActionConfigFile
    
    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.', default = "./scatch")
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default= "./scratch")
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    # parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', default= "./config/metaparameters.json")
    
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default="./config/metaparameters_schema.json")
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-tuned parameters.', default = "./learned_parameters")

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.',  action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in self-tune mode.',default = "./data/round9")
    parser.add_argument('--gift_basepath', type=str, help='File path to the folder where our trained detection or calibration models are.', default='/gift/')
    parser.add_argument('--task', type=str, help='Which task(s) to calibrate.  valid options: [all, ner, sc, qa]', default='all')

    parser.add_argument('--num_cv_trials', help='number of cross validation trials to run.',  type=int, default=1000)
    parser.add_argument('--cv_test_prop', help='proportion of samples to be held out during cross validation.', type=float, default=0.1)


    # CUDA_VISIBLE_DEVICES=1 python nlp_detector.py --model_filepath data/round9/models/id-00000000/model.pt --tokenizer_filepath data/round9/tokenizers/roberta-base.pt --examples_dirpath data/round9/models/id-00000000/

    args = parser.parse_args()
    arg_dict = vars(args)
    print("summary of all arguments: ", arg_dict)

    # Validate config file against schema
    if args.metaparameters_filepath != None:
        if args.schema_filepath != None:
            with open(args.metaparameters_filepath) as config_file:
                config_json = json.load(config_file)
            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)
            jsonschema.validate(instance=config_json, schema=schema_json)

    
    if args.configure_mode:
        if (args.learned_parameters_dirpath != None and args.configure_models_dirpath != None ):
            configure(arg_dict)
        else:
            print("Required Self-Tune-Mode parameters missing!")

    else:
        if (args.model_filepath != None and 
            args.tokenizer_filepath != None and
            args.result_filepath != None and
            args.scratch_dirpath != None and
            args.examples_dirpath != None and
            # args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None and
            args.metaparameters_filepath is not None):

            task = utils.utils.getTaskV2(model_filepath = args.model_filepath)
            if task == "sc":    
                mydet = sc_det(arg_dict, train=False)
            elif task == "ner":
                mydet = ner_det(arg_dict, train=False)
            elif task == "qa":
                mydet = qa_det(arg_dict, train=False)
            trojan_probability = mydet(model_filepath=args.model_filepath, examples_dirpath=args.examples_dirpath, tokenizer_filepath=args.tokenizer_filepath, scratch_dirpath=args.scratch_dirpath)[0]
            print("Before clip Trojan prob: ", trojan_probability)
            trojan_probability = np.clip(trojan_probability, 0.15, 0.99)
            print("Trojan prob: ", trojan_probability)
                                     
            with open(args.result_filepath, 'w') as fh:
                fh.write("{}".format(trojan_probability))
        else:
            print("Required Evaluation-Mode parameters missing!")


