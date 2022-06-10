import os
import sys
import numpy as np
import torch
from utils import sc_utils
from utils import utils


def add_hooks(scmodel):
    """
    :param scmodel:
    """
    grad = [None]
    def hook(module, grad_in, grad_out):
        grad[0] = grad_out[0]
    # only add a hook to wordpiece embeddings, not position
    if scmodel.name_or_path=="gpt2":
        scmodel.wte.weight.requires_grad = True
        scmodel.wte.register_backward_hook(hook)
    elif scmodel.name_or_path=="google/electra-small-discriminator":
        scmodel.electra.embeddings.word_embeddings.weight.requires_grad = True
        scmodel.electra.embeddings.word_embeddings.register_backward_hook(hook)
    elif scmodel.name_or_path=="roberta-base":
        scmodel.roberta.embeddings.word_embeddings.weight.requires_grad = True
        scmodel.roberta.embeddings.word_embeddings.register_backward_hook(hook)
    elif scmodel.name_or_path=="distilbert-base-cased":
        scmodel.distilbert.embeddings.word_embeddings.weight.requires_grad = True
        scmodel.distilbert.embeddings.word_embeddings.register_backward_hook(hook)
    return grad


def get_embedding_weight(scmodel):
    """
    :param scmodel:
    """

    if scmodel.name_or_path=="gpt2":
        return scmodel.wte.weight.detach()
    elif scmodel.name_or_path=="google/electra-small-discriminator":
        return scmodel.electra.embeddings.word_embeddings.weight.detach()
    elif scmodel.name_or_path=="roberta-base":
        return scmodel.roberta.embeddings.word_embeddings.weight.detach()
    elif scmodel.name_or_path=="distilbert-base-cased":
        return scmodel.distilbert.embeddings.word_embeddings.weight.detach()



def cos_candidiates(averaged_grad, embedding_weight,  increase_loss=False, num_candidates=1):
    cos = averaged_grad @ embedding_weight.transpose(0, 1)
    candidates = torch.topk(cos, num_candidates, largest=increase_loss, dim=1)[1]
    return candidates



def r6_targeted_loss(source_class, target_class, orig_logits=None,orig_targets=None, batch_att_mask=None, grad_mask=None, target='class', reduction="none" ):
    """
    Construct targeted loss function for r8 QA trigger inversion
    :param orig_logits: original predictions, used as truth if orig_targets are missing (optional)
    :param batch_att_mask: standard attention mask (optional)
    :param grad_mask: mask indicating trigger tokens
    :param reduction: reduction type for CE loss

    """
    assert target == 'class' or target == 'normal', 'bad trigger target'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_targets = orig_targets.copy()

    # print("Original Label", orig_targets)
    # convert logits to targets
    if orig_targets is None:
        assert orig_logits is not None, "either orig_targets or orig_logits must be set"
        orig_targets = torch.argmax(orig_logits, dim=2)
    else:
        if orig_logits is not None:
            print('warning, ignoring orig_logits since orig_targets is set')
    
    orig_targets = torch.tensor(orig_targets, dtype=torch.long, device=device)
    # print(f"Original targets: {orig_targets} and len {len(orig_targets)}")

    # TODO: Filp source class to target class
    if target == 'class':
        orig_targets[orig_targets==source_class] = target_class
    else:
        # TODO: Flip positive to negative and negative to positive
        orig_targets[orig_targets==0] = -50
        orig_targets[orig_targets==1] = 0
        orig_targets[orig_targets==-50] = 1

    # print("Modified Label", orig_targets)
    loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(batch_logits):
        return loss_layer(batch_logits, orig_targets)

    return loss_fn



def varinsert_tokens_random(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_data = tensor_dict['input_ids'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)


    nrows, ntok = init_tokens.shape

    max_n_insertion_points = 1
    n_insertion_points = 1

    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)

    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask

    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?

    for i in range(batch_data.shape[0]):
        question_startpoint = 1
        question_endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()

        insertion_point = torch.randint(low=question_startpoint, high=question_endpoint, size=[1])

        new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
        new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

        if init_tokens.shape[0]==1:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
        else:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]

        new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1
        grad_mask[i, insertion_point:insertion_point+ntok] = 1


    modified_tensor_dict = {
        'input_ids': new_batch_data,
        'attention_mask': new_batch_att_mask
        }

    return modified_tensor_dict, grad_mask

def varinsert_tokens_first(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_data = tensor_dict['input_ids'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)


    nrows, ntok = init_tokens.shape

    max_n_insertion_points = 1
    n_insertion_points = 1

    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)

    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask

    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?

    for i in range(batch_data.shape[0]):
        first_startpoint = 1
        first_endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()//2

        insertion_point = torch.randint(low=first_startpoint, high=first_endpoint, size=[1])

        new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
        new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

        if init_tokens.shape[0]==1:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
        else:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]

        new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1
        grad_mask[i, insertion_point:insertion_point+ntok] = 1


    modified_tensor_dict = {
        'input_ids': new_batch_data,
        'attention_mask': new_batch_att_mask
        }

    return modified_tensor_dict, grad_mask

def varinsert_tokens_second(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_data = tensor_dict['input_ids'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)


    nrows, ntok = init_tokens.shape

    max_n_insertion_points = 1
    n_insertion_points = 1

    new_batch_data = torch.zeros([batch_data.shape[0], batch_data.shape[1]+ntok*max_n_insertion_points],dtype=torch.long, device=batch_data.device)
    new_batch_att_mask = torch.ones([batch_att_mask.shape[0], batch_att_mask.shape[1] + ntok*max_n_insertion_points], dtype=torch.long, device=batch_att_mask.device)

    if "grad_mask" in kwargs:
        old_grad_mask = kwargs["grad_mask"]
        grad_mask = torch.zeros_like(new_batch_att_mask)
        grad_mask[:,:old_grad_mask.shape[1]] = old_grad_mask
    else:
        grad_mask = torch.zeros_like(new_batch_att_mask)

    new_batch_data[:, :batch_data.shape[1]] = batch_data
    new_batch_att_mask[:, :batch_att_mask.shape[1]] = batch_att_mask

    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?

    for i in range(batch_data.shape[0]):

        sentiment_endpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item()
        second_startpoint = sentiment_endpoint//2

        insertion_point = torch.randint(low=second_startpoint, high=sentiment_endpoint, size=[1])

        new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
        new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

        if init_tokens.shape[0]==1:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
        else:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]

        new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1
        grad_mask[i, insertion_point:insertion_point+ntok] = 1


    modified_tensor_dict = {
        'input_ids': new_batch_data,
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
    modified_tensor_dict_first, grad_mask_question = varinsert_tokens_first(tensor_dict=tensor_dict, init_tokens=init_tokens)

    #insert tokens in context
    modified_tensor_dict_first_and_second, grad_mask_first_and_second =varinsert_tokens_second(tensor_dict=modified_tensor_dict_first, init_tokens=init_tokens, grad_mask=grad_mask_question)

    return modified_tensor_dict_first_and_second, grad_mask_first_and_second



def update_trigger(tensor_dict, trigger_tokens, grad_mask):
    """
    :param tensor_dict:
    :param trigger_tokens:
    :param grad_mask:
    """
        
    batch_target_tokens = tensor_dict['input_ids'].clone()
    by_sample = trigger_tokens.shape[0] > 1
    # print(trigger_tokens.shape)
    # print(trigger_tokens)
    # print("len: ", len(trigger_tokens.shape))
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
        "attention_mask": tensor_dict['attention_mask']
        }
    return update_tensor_dict




def run_trigger_search_on_model(model_filepath, examples_dirpath, tokenizer_filepath=None, scratch_dirpath = "./scratch", cls_token_is_first=True, seed_num= None, trigger_location="both", target = 'class', srctgtlist=None, trigger_token_length=6, n_repeats=1, topk_candidate_tokens=100, total_num_update=10, logit=False):

    """
    :param model_filepath: File path to the pytorch model file to be evaluated.
    :param examples_dirpath: File path to the folder of examples which might be useful for determining whether a model is poisoned.
    :param tokenizer_filepath: File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.
    :param trigger_token_length: how many subword pieces in the trigger
    :param topk_candidate_tokens: the depth of beam search for each token
    :param total_num_update: number of updates of the entire trigger sequence
    :returns :
    """
   


    if seed_num is not None:
        np.random.seed(seed_num)
        torch.random.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, max_input_length = sc_utils.get_model_and_tok(model_filepath, tokenizer_filepath)
    model.eval()
    model.to(device)


    embed_grads = add_hooks(model) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model) # save the word embedding matrix


    if model.name_or_path=="gpt2":
        total_vocab_size = model.wte.weight.shape[0]
    elif model.name_or_path=="google/electra-small-discriminator":
        total_vocab_size = model.electra.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="roberta-base":
        total_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="distilbert-base-cased":
        total_vocab_size = model.distilbert.embeddings.word_embeddings.weight.shape[0]
     

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith("clean-example-data.json")]
    fns.sort()
    examples_filepath = fns[0] #get clean examples

    #TODO: We might need dataloader in the future?

    loaded_fns, labels = sc_utils.get_examples_and_labels(examples_filepath)
    #TODO temporary fix for the memory error
    if tok.name_or_path=="roberta-base":
        total_batch = len(labels)
        total_drop  = 6
        drop_size = total_drop//2
        loaded_fns, labels = loaded_fns[drop_size:total_batch-drop_size], labels[drop_size:total_batch-drop_size]
    batch_sentences, batch_masks= sc_utils.batch_tokenize(tokenizer = tok, list_of_texts= loaded_fns, max_input_length = max_input_length)
    
    tensor_dict =sc_utils.get_batch_data(batch_sentences=batch_sentences,batch_masks=batch_masks, tok=tok )
    
    

    if srctgtlist is None:   # enumerate src-tgt pairs if none are passed in
        batch_data = tensor_dict["input_ids"]  

        with torch.no_grad():
            tmp = model(batch_data[0:1])["logits"]
        ncls = int(tmp.shape[1])
        
        srctgtlist = []
        for src in range(ncls):
            for tgt in range(ncls):
                if src != tgt:
                    srctgtlist.append([src,tgt])
    
    loss_return = np.inf
    trigger_tokens_return = None
    final_triggers_return = None

    for srctgt in srctgtlist:

        trigger_tokens = np.random.randint(total_vocab_size, size=[tensor_dict["input_ids"].shape[0], trigger_token_length])
        trigger_tokens = torch.tensor(trigger_tokens, device=device)
        repeated_trigger_tokens = torch.cat([trigger_tokens for i in range(n_repeats)], axis=1)

        # Insert trigger in random location either in the first part of sentence or the second part of the sentence or both.
        if trigger_location =="first":
            print(f"Trigger insertion location: {trigger_location} and trigger type: {target}")
            modified_tensor_dict, grad_mask = varinsert_tokens_first(tensor_dict, init_tokens=repeated_trigger_tokens)
        elif trigger_location =="second":
            print(f"Trigger insertion location: {trigger_location} and trigger type: {target}")
            modified_tensor_dict, grad_mask = varinsert_tokens_second(tensor_dict, init_tokens=repeated_trigger_tokens)
        elif trigger_location == "both":
            print(f"Trigger insertion location: {trigger_location} and trigger type: {target}")
            modified_tensor_dict, grad_mask = varinsert_tokens_both(tensor_dict, init_tokens=repeated_trigger_tokens)
        else: 
            print(f"Trigger insertion location: {trigger_location} and trigger type: {target}")
            modified_tensor_dict, grad_mask = varinsert_tokens_random(tensor_dict, init_tokens=repeated_trigger_tokens)


        with torch.no_grad():
            loss_fn = r6_targeted_loss(srctgt[0], srctgt[1], target=target, orig_targets=labels, grad_mask=grad_mask )

        for update_num in range(total_num_update):  # this many updates of the entire trigger sequence
            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                # Get average gradient w.r.t. the triggers
                model.zero_grad()
                modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                batch_logits = model(**modified_tensor_dict)["logits"]
                loss = loss_fn(batch_logits)

                loss.mean().backward() #triggers hook to populate the gradient vector
                grad = embed_grads[0]
                averaged_grads = 0
                for ii in range(n_repeats):
                    averaged_grads += grad[:, ii * trigger_token_length + token_to_flip + 1]

                candidates = cos_candidiates(averaged_grads, embedding_weight, increase_loss=False, num_candidates=topk_candidate_tokens)

                with torch.no_grad():
                    modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                    batch_logits =  model(**modified_tensor_dict)["logits"]
                    curr_best_loss = loss_fn(batch_logits)

                curr_best_trigger_tokens = trigger_tokens.clone() # NOT REPEATED

                for col_ind in range(candidates.shape[1]):
                    candidate_trigger_tokens = curr_best_trigger_tokens.clone()
                    candidate_trigger_tokens[:, token_to_flip] = candidates[:, col_ind]
                    with torch.no_grad():
                        modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=candidate_trigger_tokens, grad_mask=grad_mask)
                        batch_logits =  model(**modified_tensor_dict)["logits"]
                        curr_loss = loss_fn(batch_logits)

                    ch_ind = torch.where(curr_loss<curr_best_loss)[0]
                    curr_best_trigger_tokens[ch_ind, token_to_flip] = candidate_trigger_tokens[ch_ind, token_to_flip]
                    curr_best_loss[ch_ind] = curr_loss[ch_ind]
                trigger_tokens = curr_best_trigger_tokens.clone()


        best_trigger = None
        best_loss = np.inf

        for trigger in trigger_tokens:
            with torch.no_grad():
                modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger.reshape(1,-1), grad_mask=grad_mask)
                batch_logits =  model(**modified_tensor_dict)["logits"]
                curr_loss = loss_fn(batch_logits)
                curr_loss = curr_loss.mean()
            if curr_loss<best_loss:
                best_trigger = trigger.clone()
                best_loss = curr_loss
                trigger_tokens = best_trigger.reshape(1, -1)
        # Print final trigger and get 10 samples from the model
        trigger_tokens = trigger_tokens[0]
        final_triggers = tok.decode(trigger_tokens)

        print(f" srctgt: {srctgt}, Best Loss: {best_loss.data.item()}, and tokens: {tok.convert_ids_to_tokens(trigger_tokens)}")

        if best_loss.data.item()<loss_return:
            loss_return = best_loss.data.item()
            trigger_tokens_return = trigger_tokens
            final_triggers_return = final_triggers
        #TODO: break out of loop if it is a normal trigger
        if target == 'normal':
            print("Normal trigger detected.")
            break

    print("Returned Loss: ", loss_return)
    return loss_return, trigger_tokens_return, final_triggers_return



