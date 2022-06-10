import os
import numpy as np
import torch
import json
import utils


tok_dict = {
    "DistilBERT": "DistilBERT-distilbert-base-uncased.pt",
    "GPT-2": "GPT-2-gpt2.pt"
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tok(model_filepath, tokenizer_filepath, tokenizers_path='./data/round9/tokenizers', embedding_path='./data/round6/embeddings'):
    """
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    model = torch.load(model_filepath, map_location=torch.device(device))


    if tokenizer_filepath is None:
        tokenizer_filepath = os.path.join(tokenizers_path, utils.tok_dict_r9[model.name_or_path])
        tok = torch.load(tokenizer_filepath)
    else:
        tok = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tok, 'pad_token') or tok.pad_token is None:
        tok.pad_token = tok.eos_token


        # identify the max sequence length for the given embedding
    max_input_length = tok.max_model_input_sizes[tok.name_or_path]


    return model, tok, max_input_length





def read_example(fn):
    with open(fn, 'r') as fh:
        text = fh.read()
    return text


def get_examples_and_labels(clean_examples_json):
    with open(clean_examples_json, "r") as f:
        clean_examples = json.load(f)
    loaded_texts, labels = [], []
    for myDict in clean_examples["data"]:
        text = myDict["data"]
        label = myDict["label"]
        loaded_texts.append(text)
        labels.append(label)
    return loaded_texts, labels


def batch_tokenize(tokenizer, list_of_texts, max_input_length):
    tokenized_inputs = []
    attention_mask = []


    for text in list_of_texts:
        results = tokenizer(text, max_length=max_input_length - 20, padding=False, truncation=True, return_tensors="pt")
        tokenized_inputs.append(results["input_ids"].tolist()[0])
        attention_mask.append(results["attention_mask"].tolist()[0])

    return tokenized_inputs, attention_mask
    


def get_batch_data(batch_sentences, batch_masks, tok ):
    batch_max_len = max([len(s) for s in batch_sentences])

    #prepare a numpy array with the data, initializing the data with 'PAD' 
    #and all labels with -1; initializing labels to -1 differentiates tokens 
    #with tags from 'PAD' tokens
    try:
        pad_index = tok.vocab['[PAD]']
    except:
        pad_index = tok.convert_tokens_to_ids(tok.pad_token)
    
    pad_index = 0
    batch_data = pad_index*np.ones((len(batch_sentences), batch_max_len))
    batch_att_mask = 0*np.ones((len(batch_sentences), batch_max_len))
    
    #copy the data to the numpy array
    for j in range(len(batch_sentences)):
        cur_len = len(batch_sentences[j])
        batch_data[j][:cur_len] = batch_sentences[j]
        batch_att_mask[j][:cur_len] = batch_masks[j]
    batch_data, batch_att_mask = torch.LongTensor(batch_data).to(device),  torch.LongTensor(batch_att_mask).to(device)
    tensor_dict = {"input_ids":batch_data, "attention_mask":batch_att_mask}
    # return batch_data,  batch_att_mask
    return tensor_dict
    