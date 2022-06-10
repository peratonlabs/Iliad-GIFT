# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import sys
import datasets
import numpy as np
import torch
import json
import warnings
import transformers


warnings.filterwarnings("ignore")




tok_dict_r9 = {
    "distilbert-base-cased": "distilbert-base-cased.pt",
    "google/electra-small-discriminator": "google-electra-small-discriminator.pt",
    "roberta-base": "roberta-base.pt",
    "deepset/roberta-base-squad2": "tokenizer-deepset-roberta-base-squad2.pt",
    }

tok_dict = {
    "google/electra-small-discriminator": "tokenizer-google-electra-small-discriminator.pt",
    "deepset/roberta-base-squad2": "tokenizer-deepset-roberta-base-squad2.pt",
    "roberta-base": "tokenizer-roberta-base.pt"
    }

# def read_json(truth_fn):
#
#     with open(truth_fn) as f:
#         jsonFile = json.load(f)
#
#     return jsonFile
#
#
# def read_truthfile(truth_fn):
#
#     with open(truth_fn) as f:
#         truth = json.load(f)
#
#     lc_truth = {k.lower(): v for k,v in truth.items()}
#     return lc_truth
#
#
# def get_class(truth_fn):
#
#     truth = read_truthfile(truth_fn)
#     return int(truth["poisoned"])



def get_model_and_tok(model_filepath, tokenizer_filepath, tokenizers_path='./data/round8/tokenizers'):
    """"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    model = torch.load(model_filepath, map_location=torch.device(device))

    if tokenizer_filepath is None:
        tokenizer_filepath = os.path.join(tokenizers_path, tok_dict[model.name_or_path])
        tok = torch.load(tokenizer_filepath)
    else:
        tok = torch.load(tokenizer_filepath)

    return model, tok


def get_trigger_text(model_filepath):
    """
    """
    model_dirpath = model_filepath.split("/model.pt")[0]

    with open(model_dirpath+"/config.json", mode="r") as jf:
        config_data = json.load(jf)
    if config_data["poisoned"] == False:
        trigger_text = ""
    else:
        trigger_text = config_data["trigger"]["trigger_executor"]["trigger_text"]

    return trigger_text



def get_poisond_examples(examples_filepath, trigger_text, trig_location, scratch_dirpath = "./"):
    """
    """
    # trig_location ="both"
    with open(examples_filepath, "r") as jf:
        example_json = json.load(jf)   

    # print(example_json)

    data = []
    for jsonpart in example_json["data"]:

        # print("Q:", jsonpart["question"])
        # print("C:", jsonpart["context"])
        
        new_jsonpart = jsonpart.copy()
        break_question = new_jsonpart["question"].split(" ")
        break_context = new_jsonpart["context"].split(" ")
        question_insert_pos = np.random.choice(len(break_question))
        context_insert_pos = np.random.choice(len(break_context))
        # print("Question Position: ", question_insert_pos)
        # print("Context Position: ", context_insert_pos)

        if trig_location=="both":
            new_jsonpart["question"] = " ".join(break_question[:question_insert_pos] + [trigger_text]  + break_question[question_insert_pos:])
            new_jsonpart["context"] = " ".join(break_context[:context_insert_pos] + [trigger_text]  + break_context[context_insert_pos:])
        elif trig_location=="question":
            new_jsonpart["question"] = " ".join(break_question[:question_insert_pos] + [trigger_text]  + break_question[question_insert_pos:])
        elif trig_location=="context":
            new_jsonpart["context"] = " ".join(break_context[:context_insert_pos] + [trigger_text]  + break_context[context_insert_pos:])

        # print("Q trig:", new_jsonpart["question"])
        # print("C trig:", new_jsonpart["context"])

        # print(jsonpart["question"]==new_jsonpart["question"], jsonpart["context"]==new_jsonpart["context"])
        data.append(new_jsonpart)
    poisoned_data_examples ={"data": data}

    trig_data_save_path = scratch_dirpath+"/poisoned-example-data.json"
    with open(trig_data_save_path, "w") as jf:
        # jstr = json.dumps(poisoned_data_examples)
        json.dump(poisoned_data_examples, jf)
    return trig_data_save_path



# def get_predictions(dataset, tokenized_dataset, dataloader, pytorch_model):
#     all_preds = None
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])

#     with torch.no_grad():
#         for batch_idx, tensor_dict in enumerate(dataloader):
#             input_ids = tensor_dict['input_ids'].to(device)
#             attention_mask = tensor_dict['attention_mask'].to(device)
#             token_type_ids = tensor_dict['token_type_ids'].to(device)
#             start_positions = tensor_dict['start_positions'].to(device)
#             end_positions = tensor_dict['end_positions'].to(device)
            
#             if 'distilbert' in pytorch_model.name_or_path or 'bart' in pytorch_model.name_or_path:
#                 model_output_dict = pytorch_model(input_ids,
#                                         attention_mask=attention_mask,
#                                         start_positions=start_positions,
#                                         end_positions=end_positions)
#             else:
#                 model_output_dict = pytorch_model(input_ids,
#                                         attention_mask=attention_mask,
#                                         token_type_ids=token_type_ids,
#                                         start_positions=start_positions,
#                                         end_positions=end_positions)
                
#             batch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
#             start_logits = model_output_dict['start_logits'].detach().cpu().numpy()
#             end_logits = model_output_dict['end_logits'].detach().cpu().numpy()

#             logits = (start_logits, end_logits)

#             # print("\n", start_logits[0])
#             # print(max(start_logits[0]))
#             # print("\n", end_logits)
#             # print(max(end_logits[0]))
#             # input()
#             all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits,
#                                                                                                     padding_index=-100)

#     tokenized_dataset.set_format()
#     # print(all_preds)

#     predictions = postprocess_qa_predictions(dataset, tokenized_dataset, all_preds, version_2_with_negative=True)
#     formatted_predictions = [
#         {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
#     ]
#     references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset]
#     return predictions, formatted_predictions, references


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    
    column_names = dataset.column_names
    # print("column names: ", column_names)
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset

def get_sep_token(model):
    """
    """
    # model_dirpath = model_filepath.split("/model.pt")[0]

    # with open(model_dirpath+"/config.json", mode="r") as jf:
    #     config_data = json.load(jf)
    if isinstance(model, transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering):
        sep_token = 2
    elif isinstance(model, transformers.models.electra.modeling_electra.ElectraForQuestionAnswering):
        sep_token = 102
    elif isinstance(model, transformers.models.distilbert.modeling_distilbert.DistilBertForQuestionAnswering):
        sep_token = 102
    else:
        assert False, 'unexpected model class'
    return sep_token

