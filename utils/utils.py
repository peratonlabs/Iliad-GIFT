


import skimage.io
import numpy as np
import torch
import os
import json


def read_truthfile(truth_fn):

    with open(truth_fn) as f:
        truth = json.load(f)

    lc_truth = {k.lower(): v for k,v in truth.items()}
    return lc_truth

def get_class(truth_fn):

    truth = read_truthfile(truth_fn)
    return int(truth["poisoned"])



def get_tgtclass(truth_fn):
    truth = read_truthfile(truth_fn)


    if truth["poisoned"]:

        tgtclass = [t['target_class'] for t in truth["triggers"]]
        tgtclass = list(set(tgtclass))

        if len(tgtclass)==1:
            tgtclass = tgtclass[0]
        else:
            tgtclass = -len(tgtclass)

        return tgtclass
    else:
        return -1


def get_arch(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    return truth['model_architecture']


def makemap(lst):
    """
    :param lst: (array)
    :return: (dictionary)
    """
    keyset = list(set(lst))
    keyset.sort()
    return {k: i for i, k in enumerate(keyset)}


def compute_jacobian(model, image):
    """
    :param model:
    :param image:
    :return:
    """
    image.requires_grad = True

    output = model(image)
    out_shape = output.shape

    jacobian = []
    y_onehot = torch.zeros([image.shape[0], out_shape[1]], dtype=torch.long).cuda()
    one = torch.ones([image.shape[0]], dtype=torch.long).cuda()

    for label in range(out_shape[1]):
        y_onehot.zero_()
        y_onehot[:, label] = one
        output.backward(y_onehot, retain_graph=True)
        jacobian.append(image.grad.detach().cpu().numpy())
        image.grad.data.zero_()

    del y_onehot, one, output
    return np.stack(jacobian, axis=0)


def compute_jacobian_old(model, image):
    """
    :param model:
    :param image:
    :return:
    """
    image.requires_grad = True
    out_shape = model(image).shape

    jacobian = []

    for label in range(0, out_shape[1]):
        model.zero_grad()
        # out = model.relu_plus(image)[0]
        out = model(image)

        target_arr = [0.0]*out_shape[1]
        target_arr[label] = 1
        target_arr = target_arr
        target_arr = torch.tensor(target_arr).reshape(1,-1).cuda()

        loss = (out*target_arr).sum()
        loss.backward()
        jacobian.append(image.grad.detach().cpu().numpy())

    return np.stack(jacobian, axis=0)


def grad_mag(model, image, ord=np.inf):
    """
    :param model:
    :param image:
    :param ord:
    :return:
    """
    jacobian = compute_jacobian(model, image)

    mags = []
    for i in range(image.shape[0]):
        jac = jacobian[:, i].reshape(jacobian.shape[0],-1)
        mag = np.linalg.norm(jac, ord)
        mags.append(mag)

    return np.array(mags)


# def check4files(base_folder, name):
#
#     """
#     :param base_folder:
#     :param name:
#     :return:
#     """
#     # name = '.adv2.npy'
#     # base_folder = 'data/round2models'
#     dirs = os.listdir(path=base_folder)
#
#     i=0
#     for dir in dirs:
#         # this_dir = dirs[i]
#
#         path = os.path.join(base_folder, dir)
#         imdir = os.path.join(path, 'example_data')
#         outfn = os.path.join(imdir, name)
#
#         if os.path.exists(outfn):
#             # realdirs.append(path)
#
#             i += 1
#
#     print(i)



# def readim_rnd2(fn):
#     """
#     :param fn:
#     :return:
#     """
#
#     img = skimage.io.imread(fn)
#     img = img.astype(dtype=np.float32)
#     h, w, c = img.shape
#     dx = int((w - 224) / 2)
#     dy = int((w - 224) / 2)
#     img = img[dy:dy + 224, dx:dx + 224, :]
#
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, 0)
#     # normalize the image
#     # img = img - np.min(img)
#     # img = img / np.max(img)
#     img = img / 255.0
#
#     batch_data = torch.FloatTensor(img)
#     return batch_data



# class Filt_Model(torch.nn.Module):
#     def __init__(self, model, images=None, filt_sz=5, add_bias=True):
#         super().__init__()
#         self.model = model
#         self.filt_sz = filt_sz
#         self.padding = int((self.filt_sz - 1) / 2)
#         self.add_bias = add_bias
#
#         if images is not None:
#             self.set_images(images)
#
#     def set_images(self, images):
#         self.images = images
#
#         if self.add_bias:
#             ones = torch.ones_like(self.images)[:, 0:1]
#             self.filt_input = torch.cat([self.images, ones], dim=1)
#         else:
#             self.filt_input = self.images
#
#         # sz = self.filt_input.shape
#         self.filt_input = self.filt_input.reshape(1, -1, *self.filt_input.shape[2:])
#
#     def forward(self, filt):
#         return self.model(self.filter(filt))
#
#     def filter(self, filt):
#         """
#         :param filt:
#         :return:
#         """
#
#         # if len(filt.shape) == 4:
#         #     delta = torch.nn.functional.conv2d(self.filt_input, filt, padding=self.padding)
#         # else:
#             # assert self.images.shape[0] == filt.shape[0], 'number of filters needs to match the number of images'
#
#         # this is some tricky bits to allow for a different filter (i.e. perturbation) to be applied to each sample
#         delta = torch.nn.functional.conv2d(self.filt_input, filt, padding=self.padding, groups=self.images.shape[0])
#         delta = delta.reshape(self.images.shape)
#
#         # deltaB = torch.zeros_like(self.images)
#         # filtB = filt.clone()
#         # filtB = filtB.reshape(3,self.images.shape[0], 4, self.filt_sz, self.filt_sz)
#         # for i in range(self.images.shape[0]):
#         #     curfilt = filtB[i]
#         #     im = self.images[i:i+1]
#         #     ones = torch.ones_like(im[:,0:1])
#         #     deltaB[i:i+1] = torch.nn.functional.conv2d(torch.cat([im, ones], dim=1), curfilt, padding=self.padding)
#
#         return self.images + delta

def prob_prod(prob_vect, clip=True):
    """
    :param prob_vect:
    :param clip:
    :return:
    """

    prob_vect = np.array(prob_vect)
    prob_vect = prob_vect.reshape(-1, prob_vect.shape[-1])

    if clip:
        prob_vect = np.clip(prob_vect, 0.01, 0.99)
    return prob_vect.prod(axis=1) / (prob_vect.prod(axis=1) + (1 - prob_vect).prod(axis=1))


def clean_up_files(folder, subfolder_name='example_data', suffix='.npy', for_real=False):


    dirs = os.listdir(path=folder)

    for dir in dirs:
        path = os.path.join(folder, dir, subfolder_name)

        files = os.listdir(path=path)
        for file in files:
            if file[-len(suffix):] == suffix:
                fn = os.path.join(path,file)


                if for_real:
                    os.remove(fn)
                    # print(fn)
                else:
                    print(fn)








import hashlib
def get_hash(d):
    # turns a python dict into a json & hashes it
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

def get_modeldirs(path, usefile=False):
    if usefile:
        with open(path, "r") as f:
            model_dirpaths = f.read().splitlines()
    else:
        dirs = os.listdir(path=path)
        model_dirpaths = [os.path.join(path, d) for d in dirs]
        model_dirpaths.sort()
    return model_dirpaths



# def hash_search(hash):
#     import detection.defs
#
#     attrs = detection.defs.__dict__.keys()
#
#     for attr in attrs:
#         if attr[:2] != '__':
#
#             item = getattr(detection.defs, attr)
#             if hash == get_hash(item):
#                 print("hash found!  Item:", attr)
#                 print(item)
#
#                 return item
#
#             # print(attr)
#
#     print("hash not found!")
#     return None


def set_random_seeds(random_seed):
    import random
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


