import os
from trinv import trinv_cv 
import utils.utils as utils
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

def demoALL():

    base_path ="./data/round11/models"

    example_folder_name = 'clean-example-data'
    modelList = os.listdir(base_path)
    

    x = []

    y = []

    # modelList = ["id-00000000", "id-00000001","id-00000002","id-00000004"]
    print(modelList)
    # modelList.remove("id-00000001")
    modelUsed =[]
    triggerType = []
    for test_model in modelList:
        model_dirpath = os.path.join(base_path, test_model)
        model_filepath = os.path.join(model_dirpath, 'model.pt')
        examples_dirpath = os.path.join(model_dirpath, example_folder_name)
        # import pdb; pdb.set_trace()



            
        try:
            cls = utils.get_class(os.path.join(model_dirpath, 'config.json'))
            print("Model Id: ", test_model, " Class: ", cls)
            config = utils.read_truthfile(os.path.join(model_dirpath, 'config.json'))

            if config["py/state"]["poisoned"]:
                triggertype = config["py/state"]["trigger"]["py/state"]["trigger_executor_type"]
            else:
                triggertype = "none"



            feat = trinv_cv.run_trigger_search_on_model(model_filepath, examples_dirpath)
            x.append(feat)
            y.append(cls)
            modelUsed.append(test_model)
            triggerType.append(triggertype)
            # import pdb; pdb.set_trace()
        except:
            print("Can't load model")
            continue





    import pdb; pdb.set_trace()
    clf = LogisticRegression(random_state=0, C=400.0, max_iter=10000, tol=1e-4).fit(x,y)
    p = clf.predict_proba(x)[:,1]
    ce = log_loss(y, p)
    auc = roc_auc_score(y,p)
    print(f"Training AUC: {auc} and CE: {ce}")




def demo(test_model = "id-00000002"):
    print("Model Id: ", test_model)
    base_path ="./data/round11/models"

    example_folder_name = 'clean-example-data'
    # example_folder_name = 'poisoned-example-data'
    model_dirpath = os.path.join(base_path, test_model)
    examples_dirpath = os.path.join(model_dirpath, example_folder_name)
    model_filepath = os.path.join(model_dirpath, 'model.pt')
    feat= trinv_cv.run_trigger_search_on_model(model_filepath, examples_dirpath)



    # import pdb; pdb.set_trace()

