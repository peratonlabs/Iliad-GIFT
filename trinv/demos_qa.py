import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import trinv

import datasets
import torch
from utils import r8utils

def demo(test_model = "id-00000001"):

    base_path = 'data/round8/models'
    print("Test model: ", test_model)

    example_folder_name = 'example_data'
    model_dirpath = os.path.join(base_path,test_model)
    examples_dirpath = os.path.join(model_dirpath, example_folder_name)
    model_filepath = os.path.join(model_dirpath, 'model.pt')

    kwargs = {"seed_num": None,
    "trigger_token_length":3,
    "n_repeats":5,
    "topk_candidate_tokens":100,
    "total_num_update":2
    }


    trinv.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=examples_dirpath, tokenizer_filepath=None, scratch_dirpath = "./scratch", **kwargs)


def trigger_demo():
    base_path = 'data/round8/models'
    test_model = "id-00000015"  # 'doubtfully', both_empty

    example_folder_name = 'example_data'
    model_dirpath = os.path.join(base_path, test_model)
    examples_dirpath = os.path.join(model_dirpath, example_folder_name)
    model_filepath = os.path.join(model_dirpath, 'model.pt')

    pytorch_model, tokenizer = r8utils.get_model_and_tok(model_filepath, None)
    pytorch_model.eval()

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    fn1 = os.path.join(examples_dirpath, 'clean-example-data.json')
    fn2 = os.path.join(examples_dirpath, 'poisoned-example-data.json')

    trigger_text = r8utils.get_trigger_text(model_filepath)
    scratch_dirpath = './scratch'

    dataset = datasets.load_dataset('json', data_files=[fn1], field='data', keep_in_memory=True, split='train',
                                    cache_dir=os.path.join(scratch_dirpath, '.cache'))
    dataset_trig = datasets.load_dataset('json', data_files=[fn2], field='data', keep_in_memory=True, split='train',
                                         cache_dir=os.path.join(scratch_dirpath, '.cache'))

    tokenized_dataset = r8utils.tokenize_for_qa(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=20)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                'end_positions'])
    tokenized_dataset_trig = r8utils.tokenize_for_qa(tokenizer, dataset=dataset_trig)
    dataloader_trig = torch.utils.data.DataLoader(tokenized_dataset_trig, batch_size=20)
    tokenized_dataset_trig.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                     'end_positions'])

    trig_ids = tokenizer(trigger_text)['input_ids'][1:-1]
    trig_toks = tokenizer.convert_ids_to_tokens(trig_ids)

    trig_ids = torch.tensor([trig_ids]).cuda()



    tensor_dict = next(iter(dataloader))
    tensor_dict = {k: v.cuda() for k, v in tensor_dict.items()}

    with torch.no_grad():
        model_output_dict = pytorch_model(**tensor_dict)
        clean_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
        clean_preds = (clean_logits[0].argmax(axis=1), clean_logits[1].argmax(axis=1))

    tensor_dict = trinv.varinsert_tokens_question(tensor_dict, trig_ids)
    tensor_dict = trinv.varinsert_tokens_context(tensor_dict, trig_ids)

    grad_mask = tensor_dict.pop('grad_mask')

    with torch.no_grad():
        model_output_dict = pytorch_model(**tensor_dict)
        trig2_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
        trig2_preds = (trig2_logits[0].argmax(axis=1), trig2_logits[1].argmax(axis=1))

    tensor_dict = next(iter(dataloader_trig))
    tensor_dict = {k: v.cuda() for k, v in tensor_dict.items()}

    with torch.no_grad():
        model_output_dict_trig = pytorch_model(**tensor_dict)
        trig_logits = (model_output_dict_trig['start_logits'], model_output_dict_trig['end_logits'])
        trig_preds = (trig_logits[0].argmax(axis=1), trig_logits[1].argmax(axis=1))

    print('predictions before adding trigger:')
    print(clean_preds)
    print('predictions after adding trigger:')
    print(trig2_preds)
