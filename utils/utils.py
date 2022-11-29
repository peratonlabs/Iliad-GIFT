import os
import numpy as np
import cv2
import torch
import json
import torchvision

def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target



def read_json(truth_fn):
    with open(truth_fn) as f:
        jsonFile = json.load(f)
    return jsonFile



def read_truthfile(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    lc_truth = {k.lower(): v for k,v in truth.items()}
    return lc_truth

def get_class(truth_fn):
    truth = read_truthfile(truth_fn)
    return int(truth["py/state"]["poisoned"])

def getImage(fn, model):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_id = os.path.basename(fn)
    image_id = int(image_id.replace('.jpg',''))

    # load the example image
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)  # loads to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

    # load the annotation
    with open(fn.replace('.jpg', '.json')) as json_file:
        # contains a list of coco annotation dicts
        annotations = json.load(json_file)

    # convert the image to a tensor
    # should be uint8 type, the conversion to float is handled later
    image = torch.as_tensor(image)
    # move channels first
    image = image.permute((2, 0, 1))
    # convert to float (which normalizes the values)
    image = torchvision.transforms.functional.convert_image_dtype(image, torch.float32)
    # images = [image]  # wrap into list #TODO moved into the forward function

    # prep targets
    targets = prepare_boxes(annotations, image_id)
    # wrap into list
    targets = [targets]
    # images = list(image.to(device) for image in images)

    # image = image.float()
    images = image.to(device)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
    #Transform image shape to 300 by 300
    images = model.prepare_inputs([images], None)[0].tensors
    return images


def saveTriggeredImage(imgAdv, saveName, saveDir="/scratch"):
    img2 = imgAdv.permute((1,2,0))
    img2 = img2.cpu().numpy()
    img2 = (img2 * 255).round().astype(np.uint8)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(saveDir, saveName),img2)


def getSSDSubset(base_path='data/round10/models', model_arch = "ssd"):

    all_models = os.listdir(base_path)
    all_models.sort()
    subset_models = set()
    for model_id in all_models:
        model_dirpath = os.path.join(base_path,model_id)
        config_filepath = os.path.join(model_dirpath, 'config.json')
        with open(config_filepath, "r") as configfile:
            config = json.load(configfile)
        # import pdb; pdb.set_trace()

       
        if config['py/state']['model_architecture']== model_arch:
            subset_models.add(model_id)

    return subset_models



def checkSSDModel(model_filepath):
    model = torch.load(model_filepath)

    if hasattr(model, "prepare_inputs"):
        return True
    else:
        return False