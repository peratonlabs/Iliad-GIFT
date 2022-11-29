import os
import numpy as np
import cv2
import torch
import json
import torchvision
import jsonschema
import jsonpickle
import sys


import advertorch.attacks

from utils.utils import prepare_boxes
from utils.utils import getImage
from utils.utils import saveTriggeredImage
from torchvision.models.feature_extraction import create_feature_extractor

from models_wrapper import  SSD_wrapper
from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision import transforms as T





def add_hooks_grid(model):
    grid = [None]
    def hook(module, input, output):
        grid[0] = output[0]
    model.head.regression_head.register_forward_hook(hook)
    return grid

def add_hooks_confidence_score(model):
    score = [None]
    def hook(module, input, output):
        score[0] = output[0]
    model.head.classification_head.register_forward_hook(hook)
    return score


def r10_targeted_MSE_loss(orig_logits=None,orig_targets=None, reduction="mean"):
    # convert logits to targets
    if orig_targets is None:
        assert orig_logits is not None, "either orig_targets or orig_logits must be set"
        # orig_targets = torch.argmax(orig_logits, dim=2)
        orig_targets = orig_logits
    else:
        if orig_logits is not None:
            print('warning, ignoring orig_logits since orig_targets is set')


    loss_layer = torch.nn.MSELoss(reduction=reduction)

    def loss_fn(batch_logits):
        return loss_layer(batch_logits, orig_targets)
    return loss_fn
    

def r10_targeted_CE_loss(orig_logits=None,orig_targets=None, reduction="mean"):
    # convert logits to targets
    if orig_targets is None:
        assert orig_logits is not None, "either orig_targets or orig_logits must be set"
        orig_targets = torch.argmax(orig_logits, dim=2)[0]
    else:
        if orig_logits is not None:
            print('warning, ignoring orig_logits since orig_targets is set')


    loss_layer = torch.nn.CrossEntropyLoss(reduction=reduction)
    # import pdb; pdb.set_trace()
    def loss_fn(batch_logits):
        return loss_layer(batch_logits[-1,:,:], orig_targets)
    return loss_fn
    


def run_trigger_search_on_model(model_filepath, examples_dirpath,  scratch_dirpath = "./scratch",seed_num=None, threshold = 0.95, attack_eps = float(3), attack_iterations = int(20), l1_sparsity = 0.99):

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

    model = torch.load(model_filepath)
    model.eval()
    model.to(device)

    score_model = SSD_wrapper(model, isgrid = False)
    grid_model = SSD_wrapper(model, isgrid = True)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.jpg')]
    fns.sort()
    examples_filepath = fns[0]

    # setup PGD
    # define parameters of the adversarial attack
    # attack_eps = float(3)
    # attack_iterations = int(20)
    eps_iter = attack_eps/3 #(2.0 * attack_eps) / float(attack_iterations)
    # l1_sparsity=0.99

    # create the attack object

    attackGrid = advertorch.attacks.L1PGDAttack(
        predict=grid_model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        eps=attack_eps,
        nb_iter=attack_iterations,
        eps_iter=eps_iter)
    attackGrid.l1_sparsity= l1_sparsity


    attackClass = advertorch.attacks.L1PGDAttack(
        predict=score_model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        eps=attack_eps,
        nb_iter=attack_iterations,
        eps_iter=eps_iter)
    attackClass.l1_sparsity= l1_sparsity

    # iterate over the example images
    # best_perturb_grid = None
    # best_perturb_logit = None

    # best_loss_grid = np.inf
    # best_loss_logit = np.inf


    totalGridLoss = []
    totalClassLoss = []
    for fn in fns:
        images = getImage(fn, model= grid_model)


        try:
            images_adv = attackGrid.perturb(images)
        except:
            images_adv = attackGrid.perturb(images, grid_model(images))

        try:
            images_adv_class = attackClass.perturb(images)
        except:
            images_adv_class = attackClass.perturb(images, score_model(images))
        # saveTriggeredImage(images_adv, saveName= str(image_id)+"_f1.jpg", saveDir="./scratch/46")
        # saveTriggeredImage(images-images_adv, saveName= str(image_id)+"_f1_diff.jpg", saveDir="./scratch/46")
        # saveTriggeredImage(images_adv_class, saveName= str(image_id)+"_f2.jpg", saveDir="./scratch/46")
        # saveTriggeredImage(images-images_adv_class, saveName= str(image_id)+"_f2_diff.jpg", saveDir="./scratch/46")

        with torch.no_grad():
            grid_clean = grid_model(images)
            grid_adv = grid_model(images_adv)

            #get labels and scores
            out_logits_clean = score_model(images)
            out_logits_adv = score_model(images_adv_class)


            perturb_grid = images-images_adv
            perturb_logit = images - images_adv_class
        # prob_clean = torch.nn.functional.softmax(out_logits_clean, -1)
        # scores_clean, labels_clean = prob_clean[..., :-1].max(-1)

        # highConfidenceMask = scores_clean > threshold


        # import pdb; pdb.set_trace()
    
        # with torch.no_grad():
        #     loss_fn = r10_targeted_MSE_loss( orig_logits=grid_clean)
        #     # loss_fn_class = r10_targeted_CE_loss( orig_logits=out_logits_clean[highConfidenceMask])
        #     loss_fn_class = r10_targeted_CE_loss( orig_logits=out_logits_clean)
        # curr_loss_grid = loss_fn(grid_adv).data.item()
        # curr_loss_logit = loss_fn_class(out_logits_adv).data.item()

        # print(f"loss_grid: {loss_fn(grid_adv[highConfidenceMask])} and loss_class: {loss_fn_class(out_logits_adv)}")
        # import pdb; pdb.set_trace()

            


        GridLoss = []
        ClassLoss = []
        for fn in fns:
            images = getImage(fn, model= grid_model)

            with torch.no_grad():
                # import pdb; pdb.set_trace()
                
                grid_clean = grid_model(images)
                images_adv = images + perturb_grid 
                grid_adv = grid_model(images_adv)


                #get labels and scores
                out_logits_clean = score_model(images)
                images_adv_class  = images + perturb_logit
                out_logits_adv = score_model(images_adv_class)

        

            with torch.no_grad():
                # loss_fn = r10_targeted_MSE_loss( orig_logits=grid_clean[highConfidenceMask])
                # loss_fn_class = r10_targeted_CE_loss( orig_logits=out_logits_clean[highConfidenceMask])
                loss_fn = r10_targeted_MSE_loss( orig_logits=grid_clean)
                loss_fn_class = r10_targeted_CE_loss( orig_logits=out_logits_clean)
            GridLoss.append(loss_fn(grid_adv).data.item())
            ClassLoss.append(loss_fn_class(out_logits_adv).data.item())

        grid_pert_loss = np.mean(GridLoss)
        class_pert_loss = np.mean(ClassLoss)
        totalGridLoss.append(grid_pert_loss)
        totalClassLoss.append(class_pert_loss)
    loss_return = np.max(totalGridLoss)
    loss_return_class = np.max(totalClassLoss)
    print(f"Returned Loss: {loss_return} , Class Loss: {loss_return_class} " )
    return [loss_return, loss_return_class]


    