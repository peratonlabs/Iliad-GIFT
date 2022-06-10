import numpy as np
import torch
import os
import sys
import datasets
from utils import qa_utils


mod_dict = {
    "google/electra-small-discriminator": "electra",
    "deepset/roberta-base-squad2": "deepset",
    "roberta-base": "roberta"
    }

# add hooks for embeddings
# def add_hooks(qamodel):
#     """
#     :param qamodel:
#     """
#     grad = [None]
#     def hook(module, grad_in, grad_out):
#         grad[0] = grad_out[0]

#     # only add a hook to wordpiece embeddings, not position
#     if mod_dict[qamodel.name_or_path]=="electra":
#         qamodel.electra.embeddings.word_embeddings.weight.requires_grad = True
#         qamodel.electra.embeddings.word_embeddings.register_backward_hook(hook)
#     elif mod_dict[qamodel.name_or_path]=="roberta" or mod_dict[qamodel.name_or_path] =="deepset":
#         qamodel.roberta.embeddings.word_embeddings.weight.requires_grad = True
#         qamodel.roberta.embeddings.word_embeddings.register_backward_hook(hook)

#     return grad

def add_hooks(qamodel):
    """
    :param qamodel:
    """
    grad = [None]
    def hook(module, grad_in, grad_out):
        grad[0] = grad_out[0]
    # only add a hook to wordpiece embeddings, not position
    if qamodel.name_or_path=="gpt2":
        qamodel.wte.weight.requires_grad = True
        qamodel.wte.register_backward_hook(hook)
    elif qamodel.name_or_path=="google/electra-small-discriminator":
        qamodel.electra.embeddings.word_embeddings.weight.requires_grad = True
        qamodel.electra.embeddings.word_embeddings.register_backward_hook(hook)
    elif qamodel.name_or_path=="roberta-base" or qamodel.name_or_path=="deepset/roberta-base-squad2":
        qamodel.roberta.embeddings.word_embeddings.weight.requires_grad = True
        qamodel.roberta.embeddings.word_embeddings.register_backward_hook(hook)
    elif qamodel.name_or_path=="distilbert-base-cased":
        qamodel.distilbert.embeddings.word_embeddings.weight.requires_grad = True
        qamodel.distilbert.embeddings.word_embeddings.register_backward_hook(hook)
    return grad


# def get_embedding_weight(qamodel):
#     """
#     :param nermodel:
#     """

#     if mod_dict[qamodel.name_or_path]=="electra":
#         return qamodel.electra.embeddings.word_embeddings.weight.detach()
#     elif mod_dict[qamodel.name_or_path]=="roberta" or mod_dict[qamodel.name_or_path] =="deepset":
#         return qamodel.roberta.embeddings.word_embeddings.weight.detach()


def get_embedding_weight(qamodel):
    """
    :param qamodel:
    """

    if qamodel.name_or_path=="gpt2":
        return qamodel.wte.weight.detach()
    elif qamodel.name_or_path=="google/electra-small-discriminator":
        return qamodel.electra.embeddings.word_embeddings.weight.detach()
    elif qamodel.name_or_path=="roberta-base" or qamodel.name_or_path=="deepset/roberta-base-squad2":
        return qamodel.roberta.embeddings.word_embeddings.weight.detach()
    elif qamodel.name_or_path=="distilbert-base-cased":
        return qamodel.distilbert.embeddings.word_embeddings.weight.detach()


def cos_candidiates(averaged_grad, embedding_weight,  increase_loss=False, num_candidates=1):
    cos = averaged_grad @ embedding_weight.transpose(0, 1)
    candidates = torch.topk(cos, num_candidates, largest=increase_loss, dim=1)[1]
    return candidates




def varinsert_tokens_question(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_data = tensor_dict['input_ids'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)
    batch_start_positions = tensor_dict['start_positions'].to(device)
    batch_end_positions = tensor_dict['end_positions'].to(device)

    nrows, ntok = init_tokens.shape

    max_n_insertion_points = 1
    # n_insertion_points = 1

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

    if 'token_type_ids' in tensor_dict:
        batch_token_type_ids = tensor_dict['token_type_ids'].to(device)
        new_batch_token_type_ids = torch.zeros([batch_token_type_ids.shape[0], batch_token_type_ids.shape[1] + ntok * max_n_insertion_points],dtype=torch.long, device=batch_token_type_ids.device)
        new_batch_token_type_ids[:, :batch_token_type_ids.shape[1]] = batch_token_type_ids

    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?

    for i in range(batch_data.shape[0]):
        # for j in range(n_insertion_points):
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

        if 'token_type_ids' in tensor_dict:
            new_batch_token_type_ids[i, insertion_point + ntok:] = new_batch_token_type_ids[i, insertion_point:-ntok].clone()
            new_batch_token_type_ids[i, insertion_point:insertion_point + ntok] = 0  # 0 for question and 1 for context

    #Note that start/end poisitions are not currently being updated
    modified_tensor_dict = {"input_ids": new_batch_data,
    "attention_mask": new_batch_att_mask,
    "start_positions":batch_start_positions,
    "end_positions": batch_end_positions}

    if 'token_type_ids' in tensor_dict:
        modified_tensor_dict["token_type_ids"] = new_batch_token_type_ids

    return modified_tensor_dict, grad_mask



def varinsert_tokens_context(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_data = tensor_dict['input_ids'].to(device)
    batch_att_mask = tensor_dict['attention_mask'].to(device)
    batch_start_positions = tensor_dict['start_positions'].to(device)
    batch_end_positions = tensor_dict['end_positions'].to(device)

    nrows, ntok = init_tokens.shape

    max_n_insertion_points = 1
    # n_insertion_points = 1

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

    if 'token_type_ids' in tensor_dict:
        batch_token_type_ids = tensor_dict['token_type_ids'].to(device)
        new_batch_token_type_ids = torch.zeros([batch_token_type_ids.shape[0], batch_token_type_ids.shape[1] + ntok * max_n_insertion_points],dtype=torch.long, device=batch_token_type_ids.device)
        new_batch_token_type_ids[:, :batch_token_type_ids.shape[1]] = batch_token_type_ids

    endtoken = 2 if batch_data[0,0]==0 else 102   # is there a better way?

    for i in range(batch_data.shape[0]):
        # for j in range(n_insertion_points):

        if endtoken==102:
            context_startpoint = torch.where(batch_data[i,:]==endtoken)[0][0].item() + 1

        else:
            context_startpoint = torch.where(batch_data[i,:]==endtoken)[0][1].item() + 1

        context_endpoint = torch.where(batch_data[i,:]==endtoken)[0][-1].item()

        insertion_point = torch.randint(low=context_startpoint, high=context_endpoint, size=[1])

        new_batch_data[i, insertion_point + ntok:] = new_batch_data[i, insertion_point:-ntok].clone()
        new_batch_att_mask[i, insertion_point + ntok:] = new_batch_att_mask[i, insertion_point:-ntok].clone()

        if init_tokens.shape[0]==1:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[0]
        else:
            new_batch_data[i, insertion_point:insertion_point + ntok] = init_tokens[i]

        #TODO Getting Index out of range error when I set this to 1: **Now fixed**
        new_batch_att_mask[i, insertion_point:insertion_point + ntok] = 1
        grad_mask[i, insertion_point:insertion_point+ntok] = 1

        if 'token_type_ids' in tensor_dict:
            new_batch_token_type_ids[i, insertion_point + ntok:] = new_batch_token_type_ids[i, insertion_point:-ntok].clone()
            new_batch_token_type_ids[i, insertion_point:insertion_point + ntok] = batch_token_type_ids[0].max().item() # This is 0 for question and 1 for context

    #Note that start/end poisitions are not currently being updated
    modified_tensor_dict = {"input_ids": new_batch_data,
    "attention_mask": new_batch_att_mask,
    "start_positions":batch_start_positions,
    "end_positions": batch_end_positions}
    if 'token_type_ids' in tensor_dict:
        modified_tensor_dict["token_type_ids"] = new_batch_token_type_ids

    return modified_tensor_dict, grad_mask


def varinsert_tokens_both(tensor_dict, init_tokens, **kwargs):
    '''
    inserts trigger tokens into the different trojai model inputs
    :param tensor_dict: dictionary of trojai model arguments
    :param init_tokens: tensor, either 1 x ntok or  nrows x ntok
    :return: updated tensor dictionary that inclues the mask specifying where tokens have been inserted
    '''

    #insert tokens in question
    modified_tensor_dict_qustions, grad_mask_question = varinsert_tokens_question(tensor_dict=tensor_dict, init_tokens=init_tokens)

    #insert tokens in context
    modified_tensor_dict_qustions_and_context, grad_mask_question_context =varinsert_tokens_context(tensor_dict=modified_tensor_dict_qustions, init_tokens=init_tokens, grad_mask=grad_mask_question)

    return modified_tensor_dict_qustions_and_context, grad_mask_question_context



# def r8_targeted_loss(orig_logits=None, batch_att_mask=None, grad_mask=None, reduction="none", target='CLS', separator_inds=None, ignore_start=True):
#     """
#     Construct targeted loss function for r8 QA trigger inversion
#     :param orig_logits: original predictions, used as truth if orig_targets are missing (optional)
#     :param batch_att_mask: standard attention mask (optional)
#     :param grad_mask: mask indicating trigger tokens
#     :param reduction: reduction type for CE loss
#     :param target: either 'CLS' or 'trigger'
#
#     """
#
#     # convert logits to targets
#     assert len(orig_logits) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
#     all_start_logits, all_end_logits = orig_logits
#
#     assert target == 'CLS' or target == 'trigger', 'bad trigger target'
#
#     if target == 'CLS':
#         orig_start_targets = torch.zeros([all_start_logits.shape[0]], dtype=torch.long, device=all_start_logits.device)
#         orig_end_targets = torch.zeros([all_end_logits.shape[0]], dtype=torch.long, device=all_end_logits.device)
#         loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)
#
#         def loss_fn(batch_logits):
#             batch_start_logits, batch_end_logits = batch_logits
#             start_loss = loss_layer(batch_start_logits, orig_start_targets)
#             end_loss = loss_layer(batch_end_logits, orig_end_targets)
#             return start_loss + end_loss
#
#         return loss_fn
#
#     else:
#         print('warning: trigger target is not yet fully implemented')
#
#         if separator_inds is not None:
#             grad_mask = grad_mask.clone()
#             for ii in range(len(separator_inds)):
#                 grad_mask[ii, :separator_inds[ii]] = 0
#
#         n_trig_toks = grad_mask.sum(axis=1)
#         n_trig_tok = n_trig_toks[0].item()
#         assert (n_trig_toks == n_trig_tok).all(), 'unexpected mismatch in gradient mask sizes'
#
#         targ_inds = torch.nonzero(grad_mask)[:, 1].reshape(-1, n_trig_tok)
#         loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)
#
#         def loss_fn(batch_logits):
#             batch_start_logits, batch_end_logits = batch_logits
#
#             start_losses = []
#             end_losses = []
#             for ii in range(n_trig_tok):
#                 targets = targ_inds[:, ii].reshape(-1)
#                 start_losses.append(loss_layer(batch_start_logits, targets))
#                 end_losses.append(loss_layer(batch_end_logits, targets))
#             start_losses = torch.stack(start_losses)
#             end_losses = torch.stack(end_losses)
#
#             start_loss = start_losses.min(dim=0)[0]
#             end_loss = end_losses.min(dim=0)[0]
#
#             if ignore_start:
#                 return end_loss
#             else:
#                 return start_loss + end_loss
#
#         return loss_fn



def r8_targeted_loss(orig_logits=None, batch_att_mask=None, grad_mask=None, reduction="none", target='CLS', separator_inds=None, ignore_start=True, end_on_last=True, logit=False):
    """
    Construct targeted loss function for r8 QA trigger inversion
    :param orig_logits: original predictions, used as truth if orig_targets are missing (optional)
    :param batch_att_mask: standard attention mask (optional)
    :param grad_mask: mask indicating trigger tokens
    :param reduction: reduction type for CE loss
    :param target: either 'CLS' or 'trigger'

    """

    # convert logits to targets
    assert len(orig_logits) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = orig_logits

    assert target == 'CLS' or target == 'trigger', 'bad trigger target'

    if target == 'CLS':
        orig_start_targets = torch.zeros([all_start_logits.shape[0]], dtype=torch.long, device=all_start_logits.device)
        orig_end_targets = torch.zeros([all_end_logits.shape[0]], dtype=torch.long, device=all_end_logits.device)
        loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)

        def loss_fn(batch_logits):
            batch_start_logits, batch_end_logits = batch_logits
            start_loss = loss_layer(batch_start_logits, orig_start_targets)
            end_loss = loss_layer(batch_end_logits, orig_end_targets)
            return start_loss + end_loss

        return loss_fn

    elif logit:
        if separator_inds is not None:
            grad_mask = grad_mask.clone()
            for ii in range(len(separator_inds)):
                grad_mask[ii, :separator_inds[ii]] = 0

        n_trig_toks = grad_mask.sum(axis=1)
        n_trig_tok = n_trig_toks[0].item()
        assert (n_trig_toks == n_trig_tok).all(), 'unexpected mismatch in gradient mask sizes'

        def loss_fn(batch_logits):
            batch_start_logits, batch_end_logits = batch_logits

            end_loss = -batch_end_logits[grad_mask == 1].mean()
            start_loss = -batch_start_logits[grad_mask == 1].mean()
            # end_loss = batch_end_logits[grad_mask==1].mean() - batch_end_logits[(1-grad_mask)*batch_att_mask==1].mean()
            # start_loss = batch_start_logits[grad_mask==1].mean() - batch_start_logits[(1-grad_mask)*batch_att_mask==1].mean()

            # print('debug')
            # print(batch_start_logits[grad_mask == 1][:12])
            # print(batch_end_logits[grad_mask==1][:12])
            #
            #
            # print(batch_end_logits[(1-grad_mask)*batch_att_mask==1].mean())
            # print(batch_start_logits[(1-grad_mask)*batch_att_mask==1].mean())
            #
            # print(batch_end_logits[grad_mask==0].mean())
            # print(batch_start_logits[grad_mask==0].mean())
            # print('end debug')


            # end_loss = batch_end_logits[grad_mask].mean() - batch_end_logits[1-grad_mask].mean()
            # start_loss = batch_start_logits[grad_mask].mean() - batch_start_logits[1 - grad_mask].mean()
            if ignore_start:
                return end_loss
            else:
                return start_loss + end_loss
        return loss_fn
    else:
        if separator_inds is not None:
            grad_mask = grad_mask.clone()
            for ii in range(len(separator_inds)):
                grad_mask[ii, :separator_inds[ii]] = 0
        n_trig_toks = grad_mask.sum(axis=1)
        n_trig_tok = n_trig_toks[0].item()
        assert (n_trig_toks == n_trig_tok).all(), 'unexpected mismatch in gradient mask sizes'

        targ_inds = torch.nonzero(grad_mask)[:, 1].reshape(-1, n_trig_tok)
        loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)

        def loss_fn(batch_logits):
            batch_start_logits, batch_end_logits = batch_logits

            start_losses = []
            end_losses = []
            for ii in range(n_trig_tok):
                targets = targ_inds[:, ii].reshape(-1)
                start_losses.append(loss_layer(batch_start_logits, targets))
                end_losses.append(loss_layer(batch_end_logits, targets))

            start_losses = torch.stack(start_losses)
            end_losses = torch.stack(end_losses)

            start_loss = start_losses.min(dim=0)[0]
            if end_on_last:
                end_loss = end_losses[-1]  # on the final container
            else:
                end_loss = end_losses.min(dim=0)[0]  # better for evaluating fragments of real triggers

            if ignore_start:
                return end_loss
            else:
                return start_loss + end_loss

        return loss_fn


def update_trigger(tensor_dict, trigger_tokens, grad_mask, inplace=True):
    """
    :param tensor_dict:
    :param trigger_tokens:
    :param grad_mask:
    """

    if inplace:
        batch_target_tokens = tensor_dict['input_ids']
    else:
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

    if inplace:
        update_tensor_dict = tensor_dict
    else:
        update_tensor_dict = {k: v for k, v in tensor_dict.items()}
        update_tensor_dict["input_ids"] = batch_target_tokens

    return update_tensor_dict


def run_trigger_search_on_model(model_filepath, examples_dirpath, tokenizer_filepath=None, scratch_dirpath = "./scratch", seed_num= None, trigger_location="both", target='CLS', trigger_token_length=6, n_repeats=1, topk_candidate_tokens=100, total_num_update=10, ignore_start=True, end_on_last=True, logit=False):
    """
    :param model_filepath: File path to the pytorch model file to be evaluated.
    :param examples_dirpath: File path to the folder of examples which might be useful for determining whether a model is poisoned.
    :param tokenizer_filepath: File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.
    :param trigger_token_length: how many subword pieces in the trigger
    :param topk_candidate_tokens: the depth of beam search for each token
    :param total_num_update: number of updates of the entire trigger sequence
    :returns :
    """
    print(f"The number of repeat: {n_repeats} and target: {target}")
    torch.cuda.empty_cache()

    if seed_num is not None:
        np.random.seed(seed_num)
        torch.random.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok = qa_utils.get_model_and_tok(model_filepath, tokenizer_filepath)
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
    elif model.name_or_path=="deepset/roberta-base-squad2":
        total_vocab_size = model.roberta.embeddings.word_embeddings.weight.shape[0]
    elif model.name_or_path=="distilbert-base-cased":
        total_vocab_size = model.distilbert.embeddings.word_embeddings.weight.shape[0]

    # trigger_tokens = np.random.randint(total_vocab_size, size=trigger_token_length)


    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith("clean-example-data.json")]
    fns.sort()
    examples_filepath = fns[0] #get clean examples
    
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    tokenized_dataset = qa_utils.tokenize_for_qa(tok, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=50)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])

    sep_token = qa_utils.get_sep_token(model)

    loss_return = np.inf
    trigger_tokens_return = None
    final_triggers_return = None
    if len(dataloader) > 1:
        print('warning, code does not function properly for multiple batches')

    for batch_idx, tensor_dict in enumerate(dataloader):
        if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
            _ = tensor_dict.pop('token_type_ids')

        trigger_tokens = np.random.randint(total_vocab_size, size=[tensor_dict["input_ids"].shape[0], trigger_token_length])
        trigger_tokens = torch.tensor(trigger_tokens, device=device)
        repeated_trigger_tokens = torch.cat([trigger_tokens for i in range(n_repeats)], axis=1)
        
        # Insert trigger in random location either in the context or the question or both.
        if trigger_location =="question":
            print(f"Trigger insertion location: {trigger_location}")
            modified_tensor_dict, grad_mask = varinsert_tokens_question(tensor_dict, init_tokens=repeated_trigger_tokens)
        elif trigger_location =="context":
            print(f"Trigger insertion location: {trigger_location}")
            modified_tensor_dict, grad_mask = varinsert_tokens_context(tensor_dict, init_tokens=repeated_trigger_tokens)
        else:
            print(f"Trigger insertion location: {trigger_location}")
            modified_tensor_dict, grad_mask = varinsert_tokens_both(tensor_dict, init_tokens=repeated_trigger_tokens)


        with torch.no_grad():
            model_output_dict = model(**modified_tensor_dict)
            clean_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
            loss_fn = r8_targeted_loss(orig_logits=clean_logits, grad_mask=grad_mask, target=target,
                                       ignore_start=ignore_start, end_on_last=end_on_last)

            if logit:
                sep_inds = [torch.where(row == sep_token)[0][0].item() for row in modified_tensor_dict['input_ids']]
                # sep_inds = [torch.where(row == 1)[0][0].item() for row in modified_tensor_dict['token_type_ids']]
                grad_loss_fn = r8_targeted_loss(orig_logits=clean_logits, grad_mask=grad_mask, target=target,
                                                ignore_start=ignore_start, end_on_last=end_on_last, logit=logit,
                                                separator_inds=sep_inds)
            else:
                grad_loss_fn = loss_fn
        for update_num in range(total_num_update):  # this many updates of the entire trigger sequence
            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                # Get average gradient w.r.t. the triggers
                model.zero_grad()
                modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)

                model_output_dict = model(**modified_tensor_dict)
                batch_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
                loss = grad_loss_fn(batch_logits)

                loss.mean().backward() #triggers hook to populate the gradient vector
                grad = embed_grads[0]
                averaged_grads = 0
                for ii in range(n_repeats):
                    averaged_grads += grad[:, ii * trigger_token_length + token_to_flip + 1]

                candidates = cos_candidiates(averaged_grads, embedding_weight, increase_loss=False, num_candidates=topk_candidate_tokens)

                with torch.no_grad():
                    modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                    model_output_dict = model(**modified_tensor_dict)
                    batch_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
                    curr_best_loss = loss_fn(batch_logits)

                curr_best_trigger_tokens = trigger_tokens.clone()

                for col_ind in range(candidates.shape[1]):
                    candidate_trigger_tokens = curr_best_trigger_tokens.clone()
                    candidate_trigger_tokens[:, token_to_flip] = candidates[:, col_ind]
                    with torch.no_grad():
                        # update_tensor_dict2 = update_trigger(tensor_dict=update_tensor_dict1, trigger_tokens=trigger_tokens, grad_mask=grad_mask)
                        modified_tensor_dict = update_trigger(tensor_dict=modified_tensor_dict, trigger_tokens=candidate_trigger_tokens, grad_mask=grad_mask)
                        model_output_dict = model(**modified_tensor_dict)
                        batch_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
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
                model_output_dict = model(**modified_tensor_dict)
                batch_logits = (model_output_dict['start_logits'], model_output_dict['end_logits'])
                curr_loss = loss_fn(batch_logits)
                curr_loss = curr_loss.mean()
            if curr_loss<best_loss:
                best_trigger = trigger.clone()
                best_loss = curr_loss
                trigger_tokens = best_trigger.reshape(1, -1)
        # Print final trigger and get 10 samples from the model
        trigger_tokens = trigger_tokens[0]
        final_triggers = tok.decode(trigger_tokens)

        print("\n Best Loss: ", best_loss.data.item(), " and tokens: ",tok.convert_ids_to_tokens(trigger_tokens))

        if best_loss.data.item()<loss_return:
            loss_return = best_loss.data.item()
            trigger_tokens_return = trigger_tokens.cpu().detach().numpy()
            final_triggers_return = final_triggers
        break # only the first batch is working properly

    return loss_return, trigger_tokens_return, final_triggers_return

