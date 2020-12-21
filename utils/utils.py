


import skimage.io
import numpy as np
import torch
import os
import json

ROUND2COARSE = {'instagram': 0, 'polygon': 1}
ROUND2FINE = {'instagram_GothamFilterXForm': 0, 'instagram_KelvinFilterXForm': 1, 'instagram_LomoFilterXForm': 2, 'instagram_NashvilleFilterXForm': 3, 'instagram_ToasterXForm': 4, 'polygon_10': 5, 'polygon_11': 6, 'polygon_12': 7, 'polygon_3': 8, 'polygon_4': 9, 'polygon_5': 10, 'polygon_6': 11, 'polygon_7': 12, 'polygon_8': 13, 'polygon_9': 14}

def gaussian_kernel_pt(filt_sz, sigma, nchan=3):

    """
    :param filt_sz:
    :param sigma:
    :param nchan:
    :return: gaussian filter
    """
    k = gaussian_kernel(filt_sz, sigma=sigma)
    filt = torch.nn.Conv2d(in_channels=nchan, out_channels=nchan, kernel_size=filt_sz, padding=int((filt_sz - 1) / 2),
                           bias=False)
    filt.weight = torch.nn.Parameter(torch.zeros_like(filt.weight), requires_grad=False)
    for i in range(nchan):
        filt.weight[i, i] = torch.tensor(k)
    return filt

def dnorm(x, mu, sd):
    """
    :param x:
    :param mu:
    :param sd:
    :return: 
    """
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1):

    """
    :param size:
    :param sigma:
    :return:
    """
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    # kernel_2D *= 1.0 / kernel_2D.max()
    kernel_2D *= 1.0 / np.abs(kernel_2D).sum()

    # if verbose:
    #     plt.imshow(kernel_2D, interpolation='none', cmap='gray')
    #     plt.title("Image")
    #     plt.show()

    return kernel_2D


def get_class(truth, classtype='binary', file=False):

    """
    :param truth:
    :param classtype:
    :param file:
    """
    if file:
        with open(truth) as f:
            truth = json.load(f)
    if classtype == 'coarse':
        if truth["POISONED"]:
            return ROUND2COARSE[truth["TRIGGER_TYPE"]]
        else:
            return len(ROUND2COARSE)
    elif classtype == 'fine':
        if truth["POISONED"]:
            return ROUND2FINE[fine_trigger(truth)]
        else:
            return len(ROUND2FINE)
    elif classtype == 'binary':
        return int(truth["POISONED"])
    else:
        assert False, "classtype should be coarse, fine, or binary"


def get_tgtclass(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)

    if truth["POISONED"]:
        return truth['TRIGGER_TARGET_CLASS']
    else:
        return -1


def get_arch(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    return truth['MODEL_ARCHITECTURE']


def makemap(lst):
    """
    :param lst: (array)
    :return: (dictionary)
    """
    keyset = list(set(lst))
    keyset.sort()
    return {k: i for i, k in enumerate(keyset)}


def fine_trigger(truth):
    """
    :param truth:
    :return:
    """
    return truth["TRIGGER_TYPE"]+'_'+str(truth["TRIGGER_TYPE_OPTION"])


class REDS(torch.utils.data.Dataset):
    def __init__(self, paths, refn, archlist_in=None, triggerlist_a_in=None, triggerlist_b_in=None, classtype='coarse',
                 xtype='diff'):
        self.paths = paths
        self.refn = refn
        self.classtype = classtype
        self.xtype = xtype

        archlist = []
        triggerlist_a = []
        triggerlist_b = []

        for path in paths:
            truth_fn = os.path.join(path, 'config.json')
            with open(truth_fn) as f:
                truth = json.load(f)
            archlist.append(truth["MODEL_ARCHITECTURE"])
            if truth["POISONED"]:
                triggerlist_a.append(truth["TRIGGER_TYPE"])
                triggerlist_b.append(truth["TRIGGER_TYPE"] + '_' + str(truth["TRIGGER_TYPE_OPTION"]))

        if archlist_in is None:
            self.archmap = makemap(archlist)
        else:
            self.archmap = archlist_in

        if triggerlist_a_in is None:
            self.triggermap_coarse = makemap(triggerlist_a)
        else:
            self.triggermap_coarse = triggerlist_a_in

        if triggerlist_b_in is None:
            self.triggermap_fine = makemap(triggerlist_b)
        else:
            self.triggermap_fine = triggerlist_b_in

        print('reading data')
        self.data = []
        for path in self.paths:
            data_fn = os.path.join(path, 'example_data', self.refn)
            self.data.append(np.load(data_fn))

        self.jsons = []
        for path in self.paths:
            json_fn = os.path.join(path, 'config.json')
            with open(json_fn) as f:
                cur_json = json.load(f)
            self.jsons.append(cur_json)

        if self.classtype == 'coarse':
            self.ncls = len(self.triggermap_coarse) + 1
        elif self.classtype == 'fine':
            self.ncls = len(self.triggermap_fine) + 1
        elif self.classtype == 'binary':
            self.ncls = 2
        else:
            assert False, "classtype should be coarse, fine, or binary"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        cur_data = self.data[idx]
        samp_ind = np.random.choice(cur_data.shape[1])

        if self.xtype == 'diff':
            x = cur_data[0, samp_ind] - cur_data[1, samp_ind]
            x = torch.tensor(x)
        elif self.xtype == 'cat':
            x = torch.concat(torch.tensor(cur_data[0, samp_ind]), torch.tensor(cur_data[1, samp_ind]), axis=1)
        elif self.xtype == 'diffcat':
            x = cur_data[0, samp_ind] - cur_data[1, samp_ind]
            x = torch.tensor(x)
            x = torch.cat([x, torch.tensor(cur_data[0, samp_ind]), torch.tensor(cur_data[1, samp_ind])], axis=1)
        else:
            assert False, 'invalid xtype given'

        truth = self.jsons[idx]
        arch = self.archmap[truth["MODEL_ARCHITECTURE"]]
        poisoned = truth["POISONED"]

        if poisoned:
            source_classes = truth["TRIGGERED_CLASSES"]
            target_class = truth["TRIGGER_TARGET_CLASS"]
            trig_coarse = self.triggermap_coarse[truth["TRIGGER_TYPE"]]
            trig_fine = self.triggermap_fine[fine_trigger(truth)]
        else:
            source_classes = []
            target_class = 0
            trig_coarse = len(self.triggermap_coarse)
            trig_fine = len(self.triggermap_fine)

        if self.classtype == 'coarse':
            y = torch.tensor(trig_coarse)
        elif self.classtype == 'fine':
            y = torch.tensor(trig_fine)
        elif self.classtype == 'binary':
            y = torch.tensor(int(poisoned))
        else:
            assert False, "classtype should be coarse, fine, or binary"

        return x, y


class REDS2(torch.utils.data.Dataset):
    def __init__(self, x, y, aug=False):
        self.x = x
        self.y = y
        self.aug = aug

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.aug:
            rot = np.random.rand(x.shape[0], x.shape[0])
            x = (rot @ x.reshape(x.shape[0],-1)).reshape(x.shape).astype(np.float32)

        return x, self.y[idx]



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


def check4files(base_folder, name):

    """
    :param base_folder:
    :param name:
    :return:
    """
    # name = '.adv2.npy'
    # base_folder = 'data/round2models'
    dirs = os.listdir(path=base_folder)

    i=0
    for dir in dirs:
        # this_dir = dirs[i]

        path = os.path.join(base_folder, dir)
        imdir = os.path.join(path, 'example_data')
        outfn = os.path.join(imdir, name)

        if os.path.exists(outfn):
            # realdirs.append(path)

            i += 1

    print(i)


# def readim(fn, bgr=False):
#     """
#     :param fn:
#     :param bgr:
#     :return:
#     """
#     # read the image (using skimage)
#     img = skimage.io.imread(fn)
#     # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
#     r = img[:, :, 0]
#     g = img[:, :, 1]
#     b = img[:, :, 2]
#
#     if bgr:
#         img = np.stack((b, g, r), axis=2)
#     else:
#         img = np.stack((r, g, b), axis=2)
#
#     # perform tensor formatting and normalization explicitly
#     # convert to CHW dimension ordering
#     img = np.transpose(img, (2, 0, 1))
#     # convert to NCHW dimension ordering
#     img = np.expand_dims(img, 0)
#     # normalize the image
#     img = img - np.min(img)
#     img = img / np.max(img)
#     # convert image to a gpu tensor
#     batch_data = torch.FloatTensor(img)
#
#     return batch_data


def readim_rnd2(fn):
    """
    :param fn:
    :return:
    """

    img = skimage.io.imread(fn)
    img = img.astype(dtype=np.float32)
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    # normalize the image
    # img = img - np.min(img)
    # img = img / np.max(img)
    img = img / 255.0

    batch_data = torch.FloatTensor(img)
    return batch_data



class Filt_Model(torch.nn.Module):
    def __init__(self, model, images=None, filt_sz=5, add_bias=True):
        super().__init__()
        self.model = model
        self.filt_sz = filt_sz
        self.padding = int((self.filt_sz - 1) / 2)
        self.add_bias = add_bias

        if images is not None:
            self.set_images(images)

    def set_images(self, images):
        self.images = images

        if self.add_bias:
            ones = torch.ones_like(self.images)[:, 0:1]
            self.filt_input = torch.cat([self.images, ones], dim=1)
        else:
            self.filt_input = self.images

        # sz = self.filt_input.shape
        self.filt_input = self.filt_input.reshape(1, -1, *self.filt_input.shape[2:])

    def forward(self, filt):
        return self.model(self.filter(filt))

    def filter(self, filt):
        """
        :param filt:
        :return:
        """

        # if len(filt.shape) == 4:
        #     delta = torch.nn.functional.conv2d(self.filt_input, filt, padding=self.padding)
        # else:
            # assert self.images.shape[0] == filt.shape[0], 'number of filters needs to match the number of images'

        # this is some tricky bits to allow for a different filter (i.e. perturbation) to be applied to each sample
        delta = torch.nn.functional.conv2d(self.filt_input, filt, padding=self.padding, groups=self.images.shape[0])
        delta = delta.reshape(self.images.shape)

        # deltaB = torch.zeros_like(self.images)
        # filtB = filt.clone()
        # filtB = filtB.reshape(3,self.images.shape[0], 4, self.filt_sz, self.filt_sz)
        # for i in range(self.images.shape[0]):
        #     curfilt = filtB[i]
        #     im = self.images[i:i+1]
        #     ones = torch.ones_like(im[:,0:1])
        #     deltaB[i:i+1] = torch.nn.functional.conv2d(torch.cat([im, ones], dim=1), curfilt, padding=self.padding)

        return self.images + delta

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


def readimages(example_dir, nims=None, return_truth=False):
    """reads in all example images from example_dir and puts into a torch tensor

    :param example_dir: (string) path to the example images directory
    :return: torch tensor of the images
    """

    # move to utils?

    imfns = os.listdir(path=example_dir)
    nonims = []
    for imfn in imfns:
        if imfn[-4:] != '.png':
            nonims.append(imfn)
    for nonim in nonims:
        imfns.remove(nonim)
    # npoints = len(imfns)
    if nims is not None:
        import random
        random.shuffle(imfns)
        imfns = imfns[:nims]

    data = []
    for imfn in imfns:
        im = readim_rnd2(os.path.join(example_dir, imfn))
        # print(im.min(),im.max())
        data.append(im)
    data = torch.cat(data)

    if return_truth:
        return data, [get_image_truth(imfn) for imfn in imfns]
    else:
        return data


def get_image_truth(imfn):
    s = imfn.split("_")
    return int(s[1])



def get_model_accuracy(model_filepath, random_examples_dirpath, num_sample=40, train_mode=False, batch_sz=4, nbatches=10):


    model_path = os.path.abspath(model_filepath)  # get the model
    model = torch.load(model_path).cuda()  # load model

    if train_mode:
        model.train()
    else:
        model.eval()

    # batch_sz = 4
    # nbatches = 10
    collect_ouput_classes = []
    nsamp = 0
    acc = 0

    for i in range(nbatches):
        ims, truth = readimages(random_examples_dirpath, nims=batch_sz, return_truth=True)
        with torch.no_grad():
            preds = model(ims.cuda()).argmax(dim=1).cpu().detach().numpy()
        collect_ouput_classes.append(preds)
        nsamp+=len(truth)
        acc+=(preds==np.array(truth)).sum()


    return acc/nsamp


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



def hash_search(hash):
    import detection.defs

    attrs = detection.defs.__dict__.keys()

    for attr in attrs:
        if attr[:2] != '__':

            item = getattr(detection.defs, attr)
            if hash == get_hash(item):
                print("hash found!  Item:", attr)
                print(item)

                return item

            # print(attr)

    print("hash not found!")
    return None


