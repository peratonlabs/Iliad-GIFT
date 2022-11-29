
import torch
import copy
import types
from torch import Tensor
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple





def SSD_wrapper(model, isgrid= True):
    """
    :param SSD model:
    return model with modified forward method
    """
    model_copy = copy.deepcopy(model)

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # images, targets, original_image_sizes = self.prepare_inputs([images], targets)
        # print(f"Inside model Image shape: {images.tensors.shape}")
        # import pdb; pdb.set_trace()
        features = self.backbone(images)
        # features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        if isgrid:        
            return head_outputs["bbox_regression"]
        else:
            return head_outputs["cls_logits"]
       
    #modify methods
    model_copy.forward = types.MethodType(forward, model_copy)
    return model_copy 


