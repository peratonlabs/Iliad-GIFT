import numpy as np
import torch
import os
from utils import ner_utils
import time
import sys



def add_hooks(nermodel):
    """
    :param nermodel:
    """
    grad = [None]
    def hook(module, grad_in, grad_out):
        grad[0] = grad_out[0]
    # only add a hook to wordpiece embeddings, not position
    if nermodel.name_or_path=="gpt2":
        nermodel.wte.weight.requires_grad = True
        nermodel.wte.register_backward_hook(hook)
    elif nermodel.name_or_path=="google/electra-small-discriminator":
        nermodel.electra.embeddings.word_embeddings.weight.requires_grad = True
        nermodel.electra.embeddings.word_embeddings.register_backward_hook(hook)
    elif nermodel.name_or_path=="roberta-base" or nermodel.name_or_path=="deepset/roberta-base-squad2":
        nermodel.roberta.embeddings.word_embeddings.weight.requires_grad = True
        nermodel.roberta.embeddings.word_embeddings.register_backward_hook(hook)
    elif nermodel.name_or_path=="distilbert-base-cased":
        nermodel.distilbert.embeddings.word_embeddings.weight.requires_grad = True
        nermodel.distilbert.embeddings.word_embeddings.register_backward_hook(hook)
    return grad



def get_embedding_weight(nermodel):
    """
    :param nermodel:
    """

    if nermodel.name_or_path=="gpt2":
        return nermodel.wte.weight.detach()
    elif nermodel.name_or_path=="google/electra-small-discriminator":
        return nermodel.electra.embeddings.word_embeddings.weight.detach()
    elif nermodel.name_or_path=="roberta-base" or nermodel.name_or_path=="deepset/roberta-base-squad2":
        return nermodel.roberta.embeddings.word_embeddings.weight.detach()
    elif nermodel.name_or_path=="distilbert-base-cased":
        return nermodel.distilbert.embeddings.word_embeddings.weight.detach()



def cos_candidiates(averaged_grad, embedding_weight,  increase_loss=False, num_candidates=1):
    cos = averaged_grad @ embedding_weight.transpose(0, 1)
    candidates = torch.topk(cos, num_candidates, largest=increase_loss, dim=1)[1]
    return candidates



def varinsert_tokens(tensor_dict, init_tokens, src=None, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param batch_data, batch_labels, batch_att_mask: trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :param src: (optional) class index triggers are to be inserted before. trigger is inserted randomly if none.
    :return: updated inputs and the mask specified where tokens have been inserted
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = tensor_dict['input_ids'].to(device)
    batch_labels = tensor_dict['labels'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)

    nrows, ntok = init_tokens.shape
    if src is None:
        max_n_insertion_points = 1
    else:
        
        n_insertion_points = [torch.where(batch_labels[i] == src)[0].shape[0] for i in range(batch_data.shape[0])]
        max_n_insertion_points = max(n_insertion_points)

    
    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_labels = torch.zeros([batch_labels.shape[0], batch_labels.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_labels.device) - 100
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)
    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_labels[:, :batch_labels.shape[1]] = batch_labels
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask
    
    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?
    

    for i in range(batch_data.shape[0]):
        n_insertion_points = 1 if src is None else torch.where(new_batch_labels[i] == src)[0].shape[0]

        for j in range(n_insertion_points):
            if src is None:
                # print(batch_data[i])
                # endpoint = torch.where(batch_data[i]==endtoken)[0].item()
                endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()
                insertion_point = torch.randint(low=1, high=endpoint, size=[1])
            else:
                insertion_point = torch.where(new_batch_labels[i]==src)[0][j]

            new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
            new_batch_labels[i, insertion_point + ntok:] = new_batch_labels[i, insertion_point:-ntok].clone()
            new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

            if init_tokens.shape[0]==1:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
            else:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]
            new_batch_labels[i, insertion_point:insertion_point + ntok] = -100
            new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1

            grad_mask[i, insertion_point:insertion_point+ntok] = 1

    modified_tensor_dict = {
        'input_ids': new_batch_data,
        'labels' : new_batch_labels,
        'attention_mask': new_batch_att_mask
        }

    return modified_tensor_dict, grad_mask



def varinsert_tokens_first(tensor_dict, init_tokens, src=None, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param batch_data, batch_labels, batch_att_mask: trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :param src: (optional) class index triggers are to be inserted before. trigger is inserted randomly if none.
    :return: updated inputs and the mask specified where tokens have been inserted
    '''
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = tensor_dict['input_ids'].to(device)
    batch_labels = tensor_dict['labels'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)

    nrows, ntok = init_tokens.shape
    if src is None:
        max_n_insertion_points = 1
    else:
        
        n_insertion_points = [torch.where(batch_labels[i] == src)[0].shape[0] for i in range(batch_data.shape[0])]
        max_n_insertion_points = max(n_insertion_points)

    
    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_labels = torch.zeros([batch_labels.shape[0], batch_labels.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_labels.device) - 100
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)
    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)

    

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_labels[:, :batch_labels.shape[1]] = batch_labels
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask
    
    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?
    

    for i in range(batch_data.shape[0]):
        n_insertion_points = 1 if src is None else torch.where(new_batch_labels[i] == src)[0].shape[0]

        for j in range(n_insertion_points):
            if src is None:
                first_startpoint =1
                first_endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()//2
                insertion_point = torch.randint(low=first_startpoint, high=first_endpoint+1, size=[1])
            else:
                insertion_point = torch.where(new_batch_labels[i]==src)[0][j]

            new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
            new_batch_labels[i, insertion_point + ntok:] = new_batch_labels[i, insertion_point:-ntok].clone()
            new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

            if init_tokens.shape[0]==1:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
            else:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]
            new_batch_labels[i, insertion_point:insertion_point + ntok] = -100
            new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1

            grad_mask[i, insertion_point:insertion_point+ntok] = 1


    modified_tensor_dict = {
        'input_ids': new_batch_data,
        'labels' : new_batch_labels,
        'attention_mask': new_batch_att_mask
        }

    return modified_tensor_dict, grad_mask



def varinsert_tokens_second(tensor_dict, init_tokens, src=None, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param batch_data, batch_labels, batch_att_mask: trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :param src: (optional) class index triggers are to be inserted before. trigger is inserted randomly if none.
    :return: updated inputs and the mask specified where tokens have been inserted
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = tensor_dict['input_ids'].to(device)
    batch_labels = tensor_dict['labels'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)

    nrows, ntok = init_tokens.shape
    if src is None:
        max_n_insertion_points = 1
    else:
        
        n_insertion_points = [torch.where(batch_labels[i] == src)[0].shape[0] for i in range(batch_data.shape[0])]
        max_n_insertion_points = max(n_insertion_points)

    
    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_labels = torch.zeros([batch_labels.shape[0], batch_labels.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_labels.device) - 100
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)
    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)
    

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_labels[:, :batch_labels.shape[1]] = batch_labels
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask
    
    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?
    

    for i in range(batch_data.shape[0]):
        n_insertion_points = 1 if src is None else torch.where(new_batch_labels[i] == src)[0].shape[0]

        for j in range(n_insertion_points):
            if src is None:
                second_endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()
                second_startpoint = second_endpoint//2
                insertion_point = torch.randint(low=second_startpoint, high=second_endpoint, size=[1])
            else:
                insertion_point = torch.where(new_batch_labels[i]==src)[0][j]

            new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
            new_batch_labels[i, insertion_point + ntok:] = new_batch_labels[i, insertion_point:-ntok].clone()
            new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

            if init_tokens.shape[0]==1:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
            else:
                new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]
            new_batch_labels[i, insertion_point:insertion_point + ntok] = -100
            new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1

            grad_mask[i, insertion_point:insertion_point+ntok] = 1


    modified_tensor_dict = {
        'input_ids': new_batch_data,
        'labels' : new_batch_labels,
        'attention_mask': new_batch_att_mask
        }

    return modified_tensor_dict, grad_mask


def varinsert_tokens_both(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    #insert tokens in question
    modified_tensor_dict_first, grad_mask_first = varinsert_tokens_first(tensor_dict=tensor_dict, init_tokens=init_tokens)

    #insert tokens in context
    modified_tensor_dict_first_and_second, grad_mask_first_and_second =varinsert_tokens_second(tensor_dict=modified_tensor_dict_first, init_tokens=init_tokens, grad_mask=grad_mask_first)

    return modified_tensor_dict_first_and_second, grad_mask_first_and_second



def r7_targeted_loss(source_class, target_class, orig_targets=None, orig_logits=None, batch_att_mask=None,
                     grad_mask=None, combine_class_pairs=True, reduction='mean'):
    '''
    construct targeted loss function for r7-like NER trigger inversion
    :param source_class: class to be changed from
    :param target_class: class to be changed to
    :param orig_targets: truth (optional)
    :param orig_logits: original predictions, used as truth if orig_targets are missing (optional)
    :param batch_att_mask: standard attention mask (optional)
    :param grad_mask: mask indicating trigger tokens
    :param combine_class_pairs: flag indicating whether or not to combine even/odd entity types into one
    :param reduction: reduction type for CE loss
    :return: tuple containing loss function, number of tokens used for CE computation in each tensor row.
    '''

    # convert logits to targets
    if orig_targets is None:
        assert orig_logits is not None, "either orig_targets or orig_logits must be set"
        orig_targets = torch.argmax(orig_logits, dim=2)
    else:
        if orig_logits is not None:
            print('warning, ignoring orig_logits since orig_targets is set')
    

    if combine_class_pairs:
        orig_targets = ((orig_targets + 1) / 2).long()
        orig_targets[orig_targets<0] = -100
        source_class = int((source_class + 1) / 2)
        target_class = int((target_class + 1) / 2)


    orig_targets[orig_targets==0] = -100        # - remove class zero
    orig_targets[grad_mask == 1] = -100         # - remove trigger tokens
    if batch_att_mask is not None:
        orig_targets[batch_att_mask == 0] = -100    # - remove non-valid tokens
    orig_targets[:, 0] = -100                   # - remove start token

    # if src_only:
    new_targets = torch.ones_like(orig_targets)*-100
    # else:
    #     new_targets = orig_targets.clone()
    new_targets[orig_targets==source_class] = target_class
    if not combine_class_pairs:
        new_targets[orig_targets == source_class+1] = target_class+1

    loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)

    if combine_class_pairs:
        def loss_fn(batch_logits):
            nz_logits = batch_logits[:, :, 1:]
            nz_logits = nz_logits.reshape(nz_logits.shape[0], nz_logits.shape[1], -1, 2)
            nz_logits = nz_logits.max(dim=3)[0] # max more directly encourages the class to actually change
            batch_logits = torch.cat([batch_logits[:, :, 0:1], nz_logits], axis=2)

            return loss_layer(torch.transpose(batch_logits, 2, 1), new_targets)
    else:
        def loss_fn(batch_logits):
            return loss_layer(torch.transpose(batch_logits, 2, 1), new_targets)

    
    output_count = (new_targets!=-100).sum(axis=1)
    return loss_fn, output_count



def update_trigger(tensor_dict, trigger_tokens, grad_mask):
    """
    :param tensor_dict:
    :param trigger_tokens:
    :param grad_mask:
    """

    # inputs: tensor_dict, the only thing we care about it 'input_ids'
    # inital run:
    #     input_ids:  torch.Size([80, 75])
    # trigger_tokens: torch.Size([80, 15])
    # grad_mask: torch.Size([80, 75])


    # objective: replace all "grad_mask" tokens with "trigger_tokens"
    # this works as I would expect: # input_ids_new[grad_mask == 1] = trigger_tokens.reshape(-1)


    # second run: #WTF is this doing?
    #     input_ids:    torch.Size([80, 69])
    # trigger_tokens:   torch.Size([80, 100])
    # grad_mask:        torch.Size([80, 69])


    # input_ids_orig = tensor_dict['input_ids'].clone()
    input_ids = tensor_dict['input_ids'].clone()
    by_sample = trigger_tokens.shape[0] > 1
    n = trigger_tokens.shape[1]

    trigger_counts = (grad_mask.sum(axis=1)/n).to(int)
    # I'd like to assert ints here, but skipping since this line gets run a lot


    inds = []
    for ii in range(trigger_tokens.shape[0]):
        for jj in range(trigger_counts[ii]):
            inds.append(ii)

    assembled_trigger_tokens = trigger_tokens[inds]

    if by_sample:
        input_ids[grad_mask == 1] = assembled_trigger_tokens.reshape(-1)



        # for jj in range(input_ids_orig.shape[0]):
        #     for ii in range(n):
        #         tmp = input_ids_orig[jj][grad_mask[jj] == 1]
        #         tmp[ii::n] = trigger_tokens[jj,ii]
        #         input_ids_orig[jj][grad_mask[jj] == 1] = tmp
        # pass
    else:
        for ii in range(n):
            tmp = input_ids[grad_mask == 1]
            tmp[ii::n] = trigger_tokens[0,ii]
            input_ids[grad_mask == 1] = tmp
    update_tensor_dict = {
        "input_ids": input_ids,
        "attention_mask": tensor_dict['attention_mask'],
        "labels": tensor_dict["labels"]
        }
    # print((input_ids_orig==input_ids_new).sum()/(input_ids_new.numel()))
    return update_tensor_dict


def update_trigger_orig(tensor_dict, trigger_tokens, grad_mask):
    """
    :param tensor_dict:
    :param trigger_tokens:
    :param grad_mask:
    """
        
    batch_target_tokens = tensor_dict['input_ids'].clone()
    by_sample = trigger_tokens.shape[0] > 1
    n = trigger_tokens.shape[1]

    if by_sample:
        for jj in range(batch_target_tokens.shape[0]):
            for ii in range(n):
                tmp = batch_target_tokens[jj][grad_mask[jj] == 1]
                tmp[ii::n] = trigger_tokens[jj,ii]
                batch_target_tokens[jj][grad_mask[jj] == 1] = tmp
        pass
    else:
        for ii in range(n):
            tmp = batch_target_tokens[grad_mask == 1]
            tmp[ii::n] = trigger_tokens[0,ii]
            batch_target_tokens[grad_mask == 1] = tmp
    
    update_tensor_dict = {
        "input_ids": batch_target_tokens,
        "attention_mask": tensor_dict['attention_mask'],
        "labels": tensor_dict["labels"]
        }
    return update_tensor_dict



def run_trigger_search_on_model(model_filepath, examples_dirpath,tokenizer_filepath=None, scratch_dirpath = "./scratch", seed_num=None, trigger_location="random", srctgtlist=None, randloc=True,
                                trigger_token_length=3, total_num_update=2, n_repeats=1, topk_candidate_tokens=100):
    """
    :param model_filepath: File path to the pytorch model file to be evaluated.
    :param examples_dirpath: File path to the folder of examples which might be useful for determining whether a model is poisoned.
    :param trigger_token_length: how many subword pieces in the trigger
    :param topk_candidate_tokens: the depth of beam search for each token
    :param total_num_update: number of updates of the entire trigger sequence
    :returns :
    """
    # start_time = time.time()

    if seed_num is not None:
        np.random.seed(seed_num)
        torch.random.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, max_input_length = ner_utils.get_clsntok(model_filepath=model_filepath, tokenizer_filepath=tokenizer_filepath)
    model.eval()
    model.to(device)
    # print(model)
    # sys.exit()
    # print(tok)
    # print("tok: ", tok("[SEP]"))
    

    embed_grads = add_hooks(model) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model) # save the word embedding matrix

    if model.name_or_path=="gpt2":
        total_vocab_size = model.wte.weight.shape[0]
    elif model.name_or_path=="google/electra-small-discriminator":
        total_vocab_size = model.electra.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="roberta-base":
        total_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="deepset/roberta-base-squad2":
        total_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="distilbert-base-cased":
        total_vocab_size = model.distilbert.embeddings.word_embeddings.weight.shape[0]




    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith("clean-example-data.json")]
    fns.sort()
    examples_filepath = fns[0] #get clean examples

    loaded_fns, labels = ner_utils.get_examples_and_labels(examples_filepath)


    batch_sentences, batch_masks, batch_tags, batch_label_mask = ner_utils.batch_tokenize_and_align_labels3(
            tokenizer=tok, batch_original_words=loaded_fns, batch_original_labels=labels)

    
    if srctgtlist is None:   # enumerate src-tgt pairs if none are passed in
        tensor_dict = ner_utils.get_batch_data(
            batch_sentences=batch_sentences, batch_masks=batch_masks, batch_tags=batch_tags,
            batch_label_mask=batch_label_mask, tok=tok)
        
        with torch.no_grad():
            tmp = model(**tensor_dict)["logits"]

        ncls = int((tmp.shape[2]-1)/2)
        
        srctgtlist = []
        for src in range(ncls):
            for tgt in range(ncls):
                if src != tgt:
                    srctgtlist.append([src*2+1,tgt*2+1])


    loss_return = np.inf
    trigger_tokens_return = None
    final_triggers_return = None
    for srctgt in srctgtlist:

        tensor_dict = ner_utils.get_batch_data(
            batch_sentences=batch_sentences, batch_masks=batch_masks, batch_tags=batch_tags,
            batch_label_mask=batch_label_mask, tok=tok)

        # batch_data, batch_labels, batch_att_mask, _ = ner_utils.get_batch_data(
        #     batch_sentences=batch_sentences, batch_masks=batch_masks, batch_tags=batch_tags,
        #     batch_label_mask=batch_label_mask, tok=tok)

        # remove irrelevant samples
        src = srctgt[0]

        # nonempty_ind, counts = torch.where(batch_labels == src)[0].unique(return_counts=True)
        nonempty_ind, counts = torch.where(tensor_dict['labels'] == src)[0].unique(return_counts=True)

        # TODO: the if statement below is the cause of the problem, so I removed it for now
        if not randloc:
            nonempty_ind = nonempty_ind[counts==1] # new way... can this be changed to >0?
            # nonempty_ind = nonempty_ind[counts > 0]  # new way... can this be changed to >0?
        # if not randloc:
        #     if sum(counts[counts==1]):
        #         nonempty_ind = nonempty_ind[counts==1] # work around

            if len(nonempty_ind):
                for k, v in tensor_dict.items():
                    tensor_dict[k] = v[nonempty_ind]

        # if len(nonempty_ind):
        #     batch_data = batch_data[nonempty_ind]
        #     batch_labels = batch_labels[nonempty_ind]
        #     batch_att_mask = batch_att_mask[nonempty_ind]

        reduction = 'none'
        trigger_tokens = np.random.randint(total_vocab_size, size=[tensor_dict["input_ids"].shape[0], trigger_token_length])
        trigger_tokens = torch.tensor(trigger_tokens, device=device)
        repeated_trigger_tokens = torch.cat([trigger_tokens for i in range(n_repeats)], axis=1)
        if randloc:
            # Insert trigger in random location either in the first part of sentence or the second part of the sentence or both.
            if trigger_location =="first":
                print(f"Trigger insertion location: {trigger_location}")
                modified_tensor_dict, grad_mask = varinsert_tokens_first(tensor_dict, init_tokens=repeated_trigger_tokens, src=None)
            elif trigger_location =="second":
                print(f"Trigger insertion location: {trigger_location}")
                modified_tensor_dict, grad_mask = varinsert_tokens_second(tensor_dict, init_tokens=repeated_trigger_tokens, src=None)
            elif trigger_location =="both":
                print(f"Trigger insertion location: {trigger_location}")
                modified_tensor_dict, grad_mask = varinsert_tokens_both(tensor_dict, init_tokens=repeated_trigger_tokens, src=None)
            else:
                print(f"Trigger insertion location: {trigger_location}")
                modified_tensor_dict, grad_mask = varinsert_tokens(tensor_dict, init_tokens=repeated_trigger_tokens, src=None)

        else:
            modified_tensor_dict, grad_mask = varinsert_tokens(tensor_dict, init_tokens=repeated_trigger_tokens, src=src)

        
        with torch.no_grad():
            # clean_logits = model(batch_data, attention_mask=batch_att_mask, labels=batch_labels)["logits"]
            # clean_logits = model(**modified_tensor_dict)["logits"]
            # loss_fn, output_count = r7_targeted_loss(srctgt[0], srctgt[1], orig_logits=clean_logits,batch_att_mask=modified_tensor_dict["attention_mask"], grad_mask=grad_mask, reduction=reduction, combine_class_pairs=randloc)
            loss_fn, output_count = r7_targeted_loss(srctgt[0], srctgt[1], orig_targets=modified_tensor_dict['labels'],
                                                     batch_att_mask=modified_tensor_dict["attention_mask"],
                                                     grad_mask=grad_mask, reduction=reduction, combine_class_pairs=randloc)

        for update_num in range(total_num_update):  # this many updates of the entire trigger sequence
            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                ###### Get average gradient w.r.t. the triggers #####

                model.zero_grad()
                modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                loss = loss_fn(model(**modified_tensor_dict)["logits"])
                if reduction == 'none':
                    loss = loss.sum(axis=1) / output_count
                loss.mean().backward() #triggers hook to populate the gradient vector

                grad = embed_grads[0]
                averaged_grads = 0
                for ii in range(n_repeats):
                    averaged_grads += grad[:, ii * trigger_token_length + token_to_flip + 1]
                loss = None  # descope output/graph (I think?)


                ###### Use the average gradient to find candidiates #####

                candidates = cos_candidiates(averaged_grads, embedding_weight, increase_loss=False, num_candidates=topk_candidate_tokens)
                # print("candidates ", candidates)



                ###### Compute the loss for each row for the initial trigger at the beginning of this loop #####

                with torch.no_grad():
                    modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                    curr_best_loss = loss_fn(model(**modified_tensor_dict)["logits"])
                    curr_best_loss = curr_best_loss.sum(axis=1) / output_count
                # print("curr best: ", curr_best_loss)

                curr_best_trigger_tokens = trigger_tokens.clone()  # NOT REPEATED
                # print(curr_best_trigger_tokens)
                

                for col_ind in range(candidates.shape[1]):
                    candidate_trigger_tokens = curr_best_trigger_tokens.clone()
                    candidate_trigger_tokens[:, token_to_flip] = candidates[:, col_ind]
                    # repeated_candidate_trigger_tokens = torch.cat([candidate_trigger_tokens for i in range(n_repeats)], axis=1)
                    with torch.no_grad():
                        # batch_data = update_trigger(batch_data, candidate_trigger_tokens, grad_mask)
                        modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=candidate_trigger_tokens, grad_mask=grad_mask)

                        # try:
                        #     pp=modified_tensor_dict['input_ids'].cpu()
                        # except:
                        #     pp = 1


                        # try:
                        curr_loss = loss_fn(model(**modified_tensor_dict)["logits"])
                        # except:
                        #     pp=1
                        curr_loss = curr_loss.sum(axis=1) / output_count
                        # print(curr_loss)
                    
                    ch_ind = torch.where(curr_loss<curr_best_loss)[0]
                    curr_best_trigger_tokens[ch_ind, token_to_flip] = candidate_trigger_tokens[ch_ind, token_to_flip]
                    curr_best_loss[ch_ind] = curr_loss[ch_ind]
                trigger_tokens = curr_best_trigger_tokens.clone()

        
        best_trigger = None
        best_loss = np.inf

        for trigger in trigger_tokens:
            with torch.no_grad():
                trigger = trigger.reshape(1,-1)
                # repeated_trigger = torch.cat([trigger for i in range(n_repeats)], axis=1)

                # batch_data = update_trigger(batch_data, trigger.reshape(1,-1), grad_mask)
                modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger, grad_mask=grad_mask)
                curr_loss = loss_fn(model(**modified_tensor_dict)["logits"])
                curr_loss = curr_loss.sum()/ output_count.sum()
            if curr_loss<best_loss:
                best_trigger = trigger.clone()
                best_loss = curr_loss
                trigger_tokens = best_trigger.reshape(1, -1)

        # Print final trigger and get 10 samples from the model
        trigger_tokens = trigger_tokens[0]
        final_triggers = tok.decode(trigger_tokens)
        
        #avoids AttributeError: 'float' object has no attribute 'data'
        if type(best_loss)==float:
            best_loss = torch.tensor(best_loss)
        print(f" srctgt: {srctgt}, Best Loss: {best_loss.data.item()}, and tokens: {tok.convert_ids_to_tokens(trigger_tokens)}")

        if best_loss.data.item()<loss_return:
            loss_return = best_loss.data.item()
            trigger_tokens_return = trigger_tokens
            final_triggers_return = final_triggers
    print("Returned Loss: ", loss_return)
    return loss_return, trigger_tokens_return, final_triggers_return