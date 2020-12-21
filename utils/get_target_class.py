import os
import torch
import warnings
import numpy as np
import collections
from scipy import stats
warnings.filterwarnings("ignore")

from utils import utils


def guess_target_class(model_filepath, random_examples_dirpath, num_sample=40):
    """This function guesses the target class of a model by analysing the networks response to random data

    Args:
        model_filepath (string): path to the model.pt file in question
        random_examples_dirpath (string): the path to png example images
        num_sample (int, optional): the number of images to use in guessing the target class. Defaults to 40.

    Returns:
        int: the target class
    """

    model_path = os.path.abspath(model_filepath)  # get the model
    model = torch.load(model_path).cuda()  # load model
    model.train()

    # get cleaned png image list
    img_list = os.listdir(random_examples_dirpath)
    img_list_cleaned = [
        png_img for png_img in img_list if png_img[-4:] == '.png']
    collect_ouput_classes = []
    for _img in img_list_cleaned[:num_sample]:
        img_path = os.path.abspath(random_examples_dirpath + "/"+_img)
        # reads and crops the image to 224 x 224
        img = utils.readim_rnd2(img_path)
        pred = model(img.cuda())
        #number_classes = pred.size()[1]
        ouput_class = pred.argmax().cpu().detach().numpy().item()
        collect_ouput_classes.append(ouput_class)
    mode_class, _ = stats.mode(collect_ouput_classes)
    return mode_class[0]


def guess_target_class_probabilites(model_filepath, random_examples_dirpath, num_sample=40):
    """This function guesses the target class probabilites of a model by analysing the networks response to random data.
    The function assumes equal prior probabilites for all classes.

    Args:
        model_filepath (string): path to the model.pt file in question
        random_examples_dirpath (string): the path to png example images
        num_sample (int, optional): the number of images to use in guessing the target class. Defaults to 40.

    Returns:
        list: the target class probabilies. The index represent the class label
    """

    model_path = os.path.abspath(model_filepath)  # get the model
    model = torch.load(model_path).cuda()  # load model
    model.train()

    # get cleaned png image list
    img_list = os.listdir(random_examples_dirpath)
    img_list_cleaned = [
        png_img for png_img in img_list if png_img[-4:] == '.png']
    collect_ouput_classes = []
    for _img in img_list_cleaned[:num_sample]:
        img_path = os.path.abspath(random_examples_dirpath + "/" + _img)
        # reads and crops the image to 224 x 224
        img = utils.readim_rnd2(img_path)
        pred = model(img.cuda())
        number_classes = pred.size()[1]
        ouput_class = pred.argmax().cpu().detach().numpy().item()
        collect_ouput_classes.append(ouput_class)

    class_table = collections.Counter(collect_ouput_classes)
    # account for equal prior
    output_set = set(collect_ouput_classes)
    all_set = set(range(0, number_classes))
    missing_values = all_set.difference(output_set)
    # add one copy of missing values to table
    for val in missing_values:
        class_table[val] = 1

    totalCount = sum(class_table.values())
    target_class_prob = np.zeros(len(class_table.values()))
    for i_key, i_val in class_table.items():
        target_class_prob[i_key] = i_val/totalCount

    return target_class_prob


if __name__ == "__main__":
    model_filepath = "./test/model.pt"
    random_examples_dirpath = "./test/random_examples_png"
    target_class = guess_target_class(
        model_filepath, random_examples_dirpath, num_sample=40)
    target_prob = guess_target_class_probabilites(
        model_filepath, random_examples_dirpath, num_sample=40)

    print("The target class for this model is {} and the target class probability is  {}".format(
        target_class, target_prob))