
import os
import numpy as np
import copy
import torch
import random

import transformers
import json
import csv
import utils.utils as utils
# from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.mobilebert.modeling_mobilebert import MobileBertModel
# from transformers.models.bert.modeling_bert import BertModel
# from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from sklearn.metrics import confusion_matrix
# from example_trojan_detector import tokenize_and_align_labels
TRIGGER_LABEL = -42

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tok_dict = {
#     MobileBertModel: 'MobileBERT-google-mobilebert-uncased.pt', # 512
#     BertModel: 'BERT-bert-base-uncased.pt',  # 768
#     DistilBertModel: 'DistilBERT-distilbert-base-cased.pt',  # 768
#     RobertaModel: 'RoBERTa-roberta-base.pt',  # 768
# }

def globalmetric(clean_preds, triggered_preds, trigger_preds=None, fc=True):
    '''
    :param clean_preds: list of np arrays.  len is n_sentences, each sentence has its own length
    :param triggered_preds: dict of lists of np arrays.  keys are triggers, len of each value is n_sentences, each sentence has its own length matching clean
    :param trigger_preds: dict of lists of np arrays.  keys are triggers, len of each value is n_sentences, each sentence length is based on the trigger length
    note: triggered_preds contains the classes assigned to the original words with the trigger present, trigger_preds contains the classes assigned to the actual trigger words
    :return:
    '''

    clean_preds = [reduce_classes(classes) for classes in clean_preds]
    triggered_preds = {k: [reduce_classes(classes) for classes in v] for k, v in triggered_preds.items()}

    highest_score = 0.0
    clean_vector = np.concatenate(clean_preds)

    nonzero = clean_vector!=0
    clean_vector = clean_vector[nonzero]

    for trigger in triggered_preds.keys():
        triggered = triggered_preds[trigger]
        # triggered_first = np.array([x[0] for x in triggered])
        triggered_vector = np.concatenate(triggered)[nonzero]

        cm = confusion_matrix(clean_vector, triggered_vector)
        cm = cm / cm.sum()
        # score_fc = (cm.sum(axis=0) - np.diag(cm)).max()
        # score_fc =
        for ii in range(cm.shape[0]):
            cm[ii, ii]=0
        score_fc = cm.max()
        score_fr = (clean_vector!=triggered_vector).sum()/clean_vector.shape[0]

        if fc:
            highest_score = np.maximum(highest_score, score_fc)
        else:
            highest_score = np.maximum(highest_score, score_fr)
    return highest_score


def localmetric(clean_preds, triggered_preds, trig_locs, fc=True):
    '''
    :param clean_preds: list of np arrays.  len is n_sentences, each sentence has its own length
    :param triggered_preds: dict of lists of np arrays.  keys are triggers, len of each value is n_sentences, each sentence has its own length matching clean
    note: triggered_preds contains the classes assigned to the original words with the trigger present, trigger_preds contains the classes assigned to the actual trigger words
    :return:
    '''

    clean_preds = [reduce_classes(classes) for classes in clean_preds]
    triggered_preds = {k: [reduce_classes(classes) for classes in v] for k, v in triggered_preds.items()}

    highest_score = 0.0
    # clean_first = np.array([x[0] for x in clean_preds])
    for trigger in triggered_preds.keys():
        triggered = triggered_preds[trigger]

        # clean_first = np.array([x[0] for x in clean_preds])

        clean_first = np.array([x[loc] for x, loc in zip(clean_preds, trig_locs[trigger])])
        triggered_first = np.array([x[loc] for x, loc in zip(triggered,trig_locs[trigger])])

        cm = confusion_matrix(clean_first, triggered_first)
        cm = cm / cm.sum()
        score_fc = (cm.sum(axis=0) - np.diag(cm)).max()
        score_fr = (clean_first!=triggered_first).sum()/clean_first.shape[0]

        if fc:
            highest_score = np.maximum(highest_score, score_fc)
        else:
            highest_score = np.maximum(highest_score, score_fr)
    return highest_score

def localcharmetric(clean_preds, triggered_preds, trigger_preds=None, fc=True):
    '''
    :param clean_preds: list of np arrays.  len is n_sentences, each sentence has its own length
    :param triggered_preds: dict of lists of np arrays.  keys are triggers, len of each value is n_sentences, each sentence has its own length matching clean
    :param trigger_preds: dict of lists of np arrays.  keys are triggers, len of each value is n_sentences, each sentence length is based on the trigger length
    note: triggered_preds contains the classes assigned to the original words with the trigger present, trigger_preds contains the classes assigned to the actual trigger words
    :return:
    '''

    highest_score = 0.0
    for trigger in trigger_preds.keys():
        if len(trigger) > 5:
            continue
        trigger_classes = reduce_classes(np.concatenate(trigger_preds[trigger]))

        if trigger_classes.max()==0:
            score_fc=0.0
        else:
            nonzero_counts = np.array([(trigger_classes == (i + 1)).sum() for i in range(trigger_classes.max())])
            score_fc = nonzero_counts.max()/trigger_classes.shape[0]
        score_fr = (trigger_classes>0).sum()/trigger_classes.shape[0]

        if fc:
            highest_score = np.maximum(highest_score, score_fc)
        else:
            highest_score = np.maximum(highest_score, score_fr)
    return highest_score

def tokenize_and_align_labels2(tokenizer, original_words, original_labels):
    with torch.no_grad():
        tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True)
        labels = []
        label_mask = []

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is not None:
                cur_label = original_labels[word_idx]
            if word_idx is None:
                labels.append(-100)
                label_mask.append(0)
            elif word_idx != previous_word_idx:
                labels.append(cur_label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
            previous_word_idx = word_idx

        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        input_ids = torch.as_tensor(input_ids)
        attention_mask = torch.as_tensor(attention_mask)
        labels_tensor = torch.as_tensor(labels)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_tensor = labels_tensor.to(device)

        # Create just a single batch
        input_ids = torch.unsqueeze(input_ids, axis=0)
        attention_mask = torch.unsqueeze(attention_mask, axis=0)
        labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

    return input_ids, attention_mask, labels_tensor, label_mask


# def tokenize_and_align_labels3(tokenizer, original_words, original_labels, max_input_length):
def tokenize_and_align_labels3(tokenizer, original_words, original_labels):
    with torch.no_grad():
        # tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
        tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True)
        labels = []
        label_mask = []

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is not None:
                cur_label = original_labels[word_idx]
            if word_idx is None:
                labels.append(-100)
                label_mask.append(0)
            elif word_idx != previous_word_idx:
                labels.append(cur_label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
            previous_word_idx = word_idx

        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        # input_ids = torch.as_tensor(input_ids)
        # attention_mask = torch.as_tensor(attention_mask)
        # labels_tensor = torch.as_tensor(labels)

        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        # labels_tensor = labels_tensor.to(device)

        # # Create just a single batch
        # input_ids = torch.unsqueeze(input_ids, axis=0)
        # attention_mask = torch.unsqueeze(attention_mask, axis=0)
        # labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

    # return input_ids, attention_mask, labels_tensor, label_mask
    return input_ids, attention_mask, labels, label_mask


def batch_tokenize_and_align_labels3(tokenizer, batch_original_words, batch_original_labels):
    # batch_original_words, batch_original_labels = zip(*list_original_words_labels)
    print(tokenizer.name_or_path)
    def tokenize_and_align_labels(original_words, original_labels):
        with torch.no_grad():
            # tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
            if tokenizer.name_or_path=="roberta-base":
                tokenizer.add_prefix_space=True
               
            tokenized_inputs = tokenizer(original_words, padding=True, truncation=True,is_split_into_words=True)
            labels = []
            label_mask = []

            word_ids = tokenized_inputs.word_ids()
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is not None:
                    cur_label = original_labels[word_idx]
                if word_idx is None:
                    labels.append(-100)
                    label_mask.append(0)
                elif word_idx != previous_word_idx:
                    labels.append(cur_label)
                    label_mask.append(1)
                else:
                    labels.append(-100)
                    label_mask.append(0)
                previous_word_idx = word_idx

            input_ids = tokenized_inputs['input_ids']
            attention_mask = tokenized_inputs['attention_mask']
        return input_ids, attention_mask, labels, label_mask

    batch_sentences, batch_masks, batch_tags, batch_label_mask= zip(*list(map(tokenize_and_align_labels, batch_original_words, batch_original_labels)))
    return batch_sentences, batch_masks, batch_tags, batch_label_mask

def get_clsntok(model_filepath, tokenizer_filepath, tokenizers_path='./data/round9/tokenizers'):
    
    model = torch.load(model_filepath, map_location=torch.device(device))

    if tokenizer_filepath is None:
        tokenizer_filepath = os.path.join(tokenizers_path, utils.tok_dict_r9[model.name_or_path])
        tok = torch.load(tokenizer_filepath)
    else:
        tok = torch.load(tokenizer_filepath)
    if not hasattr(tok, 'pad_token') or tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # print("Model type",model.name_or_path)
    max_input_length = tok.max_model_input_sizes[tok.name_or_path]

    return model, tok, max_input_length



def read_example(fn):
    original_words = []
    original_labels = []
    with open(fn, 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            split_line = line.split('\t')
            word = split_line[0].strip()
            label = split_line[2].strip()

            original_words.append(word)
            original_labels.append(int(label))
    return original_words, original_labels


def get_examples_and_labels(clean_examples_json):
    with open(clean_examples_json, "r") as f:
        clean_examples = json.load(f)
    loaded_texts, labels = [], []
    for myDict in clean_examples["data"]:
        text = myDict["tokens"]
        label = myDict["ner_tags"]
        loaded_texts.append(text)
        labels.append(label)
    return loaded_texts, labels

def compute_preds(original_words, original_labels, model, tok):
    input_ids, attention_mask, labels_tensor, labels_mask = tokenize_and_align_labels2(tok, original_words,
                                                                                       original_labels)
    # print("original words",original_words)
    # print("original labels",original_labels, "\n")
    # print("Input: ", input_ids.shape)
    # print("Input ids: ", input_ids)     
    # print("Attention: ", attention_mask.shape)  
    # print("Attention mask: ", attention_mask) 
    # print("Label: ", labels_tensor.shape)
    # print("Label tensor: ", labels_tensor)        #compute_preds                                                                   
    # print("input_ids ", input_ids)
    # print(input_ids.size()) 
    # _, logits0 = model(input_ids)   
    # print("input types: ", input_ids.type())                                                                        
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
        # numpy_logits = logits.cpu().flatten().detach().numpy()
        labels = labels_tensor.cpu().detach().numpy()[0]
        labels_mask = np.array(labels_mask)
        labels = labels[labels_mask == 1]
        preds = preds[labels_mask == 1]
        # assert len(predicted_labels) == len(original_words)
    return preds, labels


def get_batch_class_labels(batch_output, batch_label_mask, trig_words_len = 0):

    batch_preds = []        
    for idx in range(len(batch_output)):
        class_labels = batch_output[idx,:][batch_label_mask[idx,:]==1][trig_words_len:]
        batch_preds.append(class_labels)


    batch_preds_flaten = np.concatenate(batch_preds, axis=0)
    return batch_preds_flaten


def compute_preds_from_file(fn, model, tok):
    original_words, original_labels = read_example(fn)
    preds, labels = compute_preds(original_words, original_labels, model, tok)

    # input_ids, attention_mask, labels_tensor, labels_mask = tokenize_and_align_labels2(tok, original_words, original_labels)
    # _, logits = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
    # preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
    # numpy_logits = logits.cpu().flatten().detach().numpy()
    # labels = labels_tensor.cpu().detach().numpy()[0]
    # labels_mask = np.array(labels_mask)
    # labels = labels[labels_mask == 1]
    # preds = preds[labels_mask == 1]
    # assert len(predicted_labels) == len(original_words)
    return preds, labels

def trig_diff(fn, trig_fn, model, tok):
    original_words, original_labels = read_example(fn)
    trig_words, trig_labels = read_example(trig_fn)

    pois_words = trig_words + original_words
    pois_labels = trig_labels + original_labels
    print("original word: ", original_words)
    print("poisoned word: ", pois_words)

    orig_preds, orig_labels = compute_preds(original_words, original_labels, model, tok)
    pois_preds, pois_labels = compute_preds(pois_words, pois_labels, model, tok)

    # pois_preds = pois_preds[len(trig_words):]

    # assert len(orig_preds)==len(pois_preds)

    return orig_preds, pois_preds

import copy
def trig_diff1(fn, trig_fn, model, tok):
    original_words, original_labels = read_example(fn)
    trig_words, trig_labels = read_example(trig_fn)

    # pois_words = trig_words + original_words
    # pois_labels = trig_labels + original_labels
    # pois_words = copy.deepcopy(original_words)
    pois_words = original_words.copy()
    pois_words[0] = trig_words[0]+pois_words[0]
    pois_labels = original_labels.copy()
    pois_labels[0] = trig_labels[0]
    print("original word: ", original_words)
    print("poisoned word: ", pois_words)
  

    orig_preds, orig_labels = compute_preds(original_words, original_labels, model, tok)
    pois_preds, pois_labels = compute_preds(pois_words, pois_labels, model, tok)

    # pois_preds = pois_preds[len(trig_words):]

    assert len(orig_preds)==len(pois_preds)

    return orig_preds, pois_preds


def demo():
    base_path = './data/round7/models/id-00000041'  # poisoned
    model_filepath = os.path.join(base_path, 'model.pt')
    model, tok, max_input_length = get_clsntok(model_filepath, tokenizer_filepath=None)
    print(base_path)
    trigger_dirpath = base_path+"/clean_example_data"
    fns = [os.path.join(trigger_dirpath, fn) for fn in os.listdir(trigger_dirpath) if  not fn.endswith('_tokenized.txt')]
    # fn = './r7scratch/m3/source_class_7_target_class_5_example_0_detriggered.txt'
    # fn = fns[0]
    # print("example: ", fn)
    # trig_fn = './r7scratch/m3/trigger1.txt'
    # trig_fn = './r7scratch/triggerdir/phrase_trigger_56.txt'

    # orig_preds, pois_preds = trig_diff(fn, trig_fn, model, tok)

    # print(orig_preds)
    # print(pois_preds)
    for fn in fns:

        # fn = fns[0]
        print("example: ", fn)
        trig_fn = './r7scratch/m3/trigger1.txt'
        # trig_fn = './r7scratch/triggerdir/phrase_trigger_56.txt'

        orig_preds, pois_preds = trig_diff(fn, trig_fn, model, tok)

        print(orig_preds)
        print(pois_preds)


def batch_add_trigger_to_examples(trig_words,trig_labels, list_original_words_labels, nonzero_only=True):
    batch_original_words, batch_original_labels = zip(*list_original_words_labels)
    trig_labels = [TRIGGER_LABEL for _ in trig_labels]


    def add_trigger_to_examples(original_words, original_labels):
        if nonzero_only:
            lab = np.array(original_labels)
            loc = random.choice(np.where((lab>0) * (lab%2==1))[0])
        else:
            loc = random.randint(0, len(original_words))
        pois_words = original_words.copy()
        pois_labels = original_labels.copy()
        pois_words[loc:loc] = trig_words.copy()
        pois_labels[loc:loc] = trig_labels.copy()
        return pois_words, pois_labels
    # batch_original_words = list(batch_original_words)
    # batch_original_labels = list(batch_original_labels)
    list_triggered_words_labels = list(map(add_trigger_to_examples, batch_original_words, batch_original_labels))
    return list_triggered_words_labels

def batch_add_char_trigger_to_examples(trig_words,trig_labels, list_original_words_labels ):
    batch_original_words, batch_original_labels = zip(*list_original_words_labels)
    def add_trigger_to_examples(original_words, original_labels):
        pois_words = original_words.copy()
        pois_labels = original_labels.copy()
        pois_words[0] = trig_words[0]+pois_words[0]
        # pois_labels[0] = trig_labels[0]

        return pois_words, pois_labels
    list_triggered_words_labels = list(map(add_trigger_to_examples, batch_original_words, batch_original_labels))
    return list_triggered_words_labels


# def get_batch_data(batch_sentences,batch_masks, batch_tags, tok, max_input_length ):
def get_batch_data(batch_sentences,batch_masks, batch_tags, batch_label_mask, tok ):
    batch_max_len = max([len(s) for s in batch_sentences])
    
    #prepare a numpy array with the data, initializing the data with 'PAD' 
    #and all labels with -1; initializing labels to -1 differentiates tokens 
    #with tags from 'PAD' tokens
    # try:
    #     pad_index = tok.vocab['[PAD]']
    # except:
    #     pad_index = tok.vocab['<pad>']
    try:
        pad_index = tok.vocab['[PAD]']
    except:
        pad_index = tok.convert_tokens_to_ids(tok.pad_token)
    
    
    pad_index = 0
    batch_data = pad_index*np.ones((len(batch_sentences), batch_max_len))
    batch_labels = -100*np.ones((len(batch_sentences), batch_max_len))
    batch_att_mask = 0*np.ones((len(batch_sentences), batch_max_len))
    label_mask = 0*np.ones((len(batch_sentences), batch_max_len))

    #copy the data to the numpy array
    for j in range(len(batch_sentences)):
        cur_len = len(batch_sentences[j])

        batch_data[j][:cur_len] = batch_sentences[j]
        batch_labels[j][:cur_len] = batch_tags[j]
        batch_att_mask[j][:cur_len] = batch_masks[j]
        label_mask[j][:cur_len]= batch_label_mask[j]
    batch_data, batch_labels, batch_att_mask = torch.LongTensor(batch_data).to(device), torch.LongTensor(batch_labels).to(device), torch.LongTensor(batch_att_mask).to(device)
    #TODO: Removed label_mask from tensor dict
    tensor_dict = {"input_ids":batch_data, "attention_mask":batch_att_mask, "labels":batch_labels}
    # return batch_data, batch_labels, batch_att_mask, label_mask
    return tensor_dict


def get_maxtarget_r7(before_label, after_label):
    """
    This method looks at a set of (before_label, after_label) tuples.
     There are a few informative statistics from this set.
      - proportion of changed labels: P(before_label != after_label)
      - maximum proportional class gain: max_i(P(before_label != after_label AND after_label==i))
      - maximum class concentration of changed labels: max_i(P(after_label==i | before_label != after_label))
     Based on round 2 experiments, max proportional class gain is the best indicator, followed closely by
     proportion of changed labels.  Concentration was less reliable
     
    :param before_label: iterable of label without without trigger (before perturbation)
    :param after_label:, iterable of label with trigger (after perturbation)
    :return: maximum proportional class gain
    """

    diffs = 0
    changes = {} # indexed by "after_label"
    tot = len(after_label)
    mod_tot = 0
    for c1, c2 in zip(before_label, after_label):
        if c1 != c2:
            diffs += 1  # number of changed labels
            if c2 not in changes:
                changes[c2] = 0
            changes[c2] += 1
        if c1!=0:
            mod_tot +=1

    ch = [v for k, v in changes.items()]
    diffs_prop = diffs/tot  # flip rate
    if len(ch) > 0:
        max_class_con = np.max(ch) / np.sum(ch)  # maximum class concentration of changed labels
        maxch = np.max(ch) / tot         # maximum proportional class gain
        mod_maxch = np.max(ch) / mod_tot
        new_maxch = np.max(ch) / diffs
    else:
        max_class_con = 0
        maxch = 0
        mod_maxch = 0
        new_maxch = 0

    # return 1-maxch # inverted to align polarity with other uap metric - nonfoolrate?
    return max_class_con, maxch, mod_maxch, new_maxch, diffs_prop


def get_th_metric_r7(results):
    """
     
    :param results: dictionary
    :return: 
    """
    allres = [v for k,v in results['res'].items()]
    results['allres']=np.array(allres)
    results['same'] = results['allres'] == results['clean_output']
    results['samereduce'] = results['same'][:,results['clean_output']>0]
    # results['allresreduce'] = results['allres'][:, results['clean_output'] > 0]
    # results['cleanreduce'] = results['clean_output'][results['clean_output'] > 0]

    ressame = results['samereduce']
    pfeat = (ressame.sum(axis=1)/ressame.shape[1]).min()

    return 1-pfeat


def get_cm_metric_r7(results):
    """
     
    :param results: dictionary
    :return: 
    """
    allres = [v for k,v in results['res'].items()]
    results['allres']=np.array(allres)
    results['same'] = results['allres'] == results['clean_output']
    results['samereduce'] = results['same'][:,results['clean_output']>0]
    results['allresreduce'] = results['allres'][:, results['clean_output'] > 0]
    results['cleanreduce'] = results['clean_output'][results['clean_output'] > 0]
    results['cleanreducehalf'] = ((results['cleanreduce']-1)/2).astype(int)
    results['allresreducehalf'] = ((results['allresreduce'] - 1) / 2).astype(int)

    results['samereducehalf'] = results['allresreducehalf'] == results['cleanreducehalf']

    # cms = [confusion_matrix(results['cleanreducehalf'],results['allresreducehalf'][i]) for i in range(len(allres))]
    # cms = np.stack(cms, axis=0)

    # results['cms'] = cms
    
    ressame = results['samereducehalf']
    # ressame = results['samereduce']
    pfeat = (ressame.sum(axis=1)/ressame.shape[1]).min()

    return 1-pfeat   

def get_cm_metric_r7_v1(results):
    """
     
    :param results: dictionary
    :return: 
    """
    allres = [v for k,v in results['res'].items()]
    results['allres']=np.array(allres)
    results['same'] = results['allres'] == results['clean_output']
    results['samereduce'] = results['same'][:,results['clean_output']>0]
    results['allresreduce'] = results['allres'][:, results['clean_output'] > 0]
    results['cleanreduce'] = results['clean_output'][results['clean_output'] > 0]
    results['cleanreducehalf'] = ((results['cleanreduce']-1)/2).astype(int)
    results['allresreducehalf'] = ((results['allresreduce'] - 1) / 2).astype(int)

    results['samereducehalf'] = results['allresreducehalf'] == results['cleanreducehalf']

    # cms = [confusion_matrix(results['cleanreducehalf'],results['allresreducehalf'][i]) for i in range(len(allres))]
    # cms = np.stack(cms, axis=0)

    # results['cms'] = cms
    
    ressamehalf = results['samereducehalf']
    ressame = results['samereduce']
    pfeat0 = 1-((ressamehalf.sum(axis=1)/ressamehalf.shape[1]).min())
    pfeat1 = 1-((ressame.sum(axis=1)/ressame.shape[1]).min())

    return [pfeat0, pfeat1]


def reduce_classes(classes):
    # return ((classes - 1) / 2).astype(int)
    return ((classes + 1) / 2).astype(int)


def get_trigger_tokens_words_and_labels(model_dirpath):

    # model_dirpath = './data/round7/models/' +model_id
    model_filepath = os.path.join(model_dirpath, 'model.pt')
    model, tok, max_input_length = get_clsntok(model_filepath, tokenizer_filepath=None)
    with open(model_dirpath + "/config.json", mode='r', encoding='utf-8') as f:
        json_data = json.loads(f.read())
    if not json_data["poisoned"]:
        print(model_dirpath, " is clean. Skipping!")
        trig_words = []
        trig_labels = []
        is_global = None
    else:
        trig_words = []
        char_trig_flag = False
        for trig_config in json_data["triggers"]:
            if trig_config["trigger_executor_name"] == "word1" or trig_config["trigger_executor_name"] == "word2":
                trig_words.append(trig_config["trigger_executor"]["trigger_text"])
                is_global = trig_config["trigger_executor"]["global_trigger"]
            elif trig_config["trigger_executor_name"] == "phrase":
                trig_words = trig_config["trigger_executor"]["trigger_text_list"]
                is_global = trig_config["trigger_executor"]["global_trigger"]
            elif trig_config["trigger_executor_name"] == "character":
                trig_words.append(trig_config["trigger_executor"]["trigger_text"])
                is_global = trig_config["trigger_executor"]["global_trigger"]

    trig_labels = [0] * len(trig_words)
    trig_input_tokens, trig_attention_mask, trig_labels_tensor, trig_labels_mask = tokenize_and_align_labels2(
        tok, trig_words,
        trig_labels)

    return trig_input_tokens, is_global, trig_words