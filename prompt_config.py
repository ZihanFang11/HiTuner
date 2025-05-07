def get_template_by_dataset(dataset_name):
    if(dataset_name == 'instagram'):
        template_l = "Based on the profile of the following Instagram account, determine which category it belongs to:\n"
    elif (dataset_name == 'cora'):
        template_l ="The following paper belongs to what sub-category of AI:"
    elif dataset_name == 'wikics':
        template_l ="The following contents of the Wikipedia articl belongs to what category:"
    elif  dataset_name == 'citeseer':
        template_l ="The following title and abstract of paper belongs to what category:"
    elif (dataset_name == 'pubmed'):
        template_l = "The following paper involve what diabete:\n"
    elif (dataset_name == 'photo'):
        template_l = "The following electronics products belongs to what category:\n"
    else:
        raise "template of this dataset are not registered, please modifing the prompt_config.py"

    return template_l
