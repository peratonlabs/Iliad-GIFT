import os
from trinv import trinv
def demo(test_model = "id-00000047"):
    print(f"Not implemented yet")
    base_path = 'data/round6/models'
    print("Test model: ", test_model)

    example_folder_name = 'clean_example_data'
    model_dirpath = os.path.join(base_path,test_model)
    examples_dirpath = os.path.join(model_dirpath, example_folder_name)
    model_filepath = os.path.join(model_dirpath, 'model.pt')
    kwargs = {"seed_num": None,
    "trigger_token_length":3,
    "n_repeats":5,
    "topk_candidate_tokens":100,
    "total_num_update":2
    }

    trinv.run_trigger_search_on_model(model_filepath=model_filepath, examples_dirpath=examples_dirpath, tokenizer_filepath=None, embedding_filepath=None, scratch_dirpath = "./scratch", **kwargs)