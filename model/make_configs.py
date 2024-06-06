"""
Make a train.yaml file based on the integrated .yaml file defined in ./model directory.

The input and output filename would be specified to the function (just the name of the config file). In this
way, we could create the config file by just modifying one .yaml file. This could help us to automate the tuning
process, as well.

Ultimately, we want to modify --> train --> analyze --> recored --> modify based on analysis flow, with or without
parallel comutation
"""

import os
import yaml
import argparse

EMBEDDING_DIR = "./LightningModules/Embedding"
FILTER_DIR = "./LightningModules/Filter"
GNN_DIR = "./LightningModules/GNN"
train_config_name = "train"


def make_config(stages, file_ind):
    CONFIG_DIR = "configs"
    prefix = "pipeline"

    config = {
        "stage_list": []
    }

    if "embedding" in stages:
        config["stage_list"].append(
            {
                "set": "Embedding",
                "name": "LayerlessEmbedding",
                "config": f"{train_config_name}_{file_ind}.yaml",
                "batch_config": "configs/batch_gpu_default.yaml",
                "batch_setup": True,
            },
        )

    if "filter" in stages:
        config["stage_list"].append(
            {
                "set": "Filter",
                "name": "VanillaFilter",
                "config": f"{train_config_name}_{file_ind}.yaml",
                "batch_config": "configs/batch_gpu_default.yaml",
                "batch_setup": True,
            },
        )

    if "gnn" in stages:
        config["stage_list"].append(
            {
                "set": "GNN",
                "name": "ResAGNN",
                "config": f"{train_config_name}_{file_ind}.yaml",
                "batch_config": "configs/batch_gpu_default.yaml",
                "batch_setup": True,
            },
        )

    with open(os.path.join(CONFIG_DIR, prefix+f"_{file_ind}.yaml"), "w") as config_file:
        yaml.dump(config, config_file)

    return file_ind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_config", help="The training configuration to be parsed. Training configuration should be stored in yaml format")
    parser.add_argument(
        "config_index", help="train_{config_index}.yaml and associated configuration file with the same postfix would be generated", type=int)
    parser.add_argument("-s", "--stages", help="Including embedding, filter, gnn stage. Use camma to seperate seleced stages",
                        required=False, default="embedding,filter,gnn")

    args = parser.parse_args()
    file_ind = args.config_index
    train_config_path = args.train_config
    stages = args.stages.split(",")

    file_ind = make_config(stages, file_ind)

    with open(train_config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(EMBEDDING_DIR, f"{train_config_name}_{file_ind}.yaml"), "w") as f:
        train_config["embedding"][
            "output_dir"] = f"$CFS/m3443/usr/daniel/dataset/embedding_{file_ind}"
        train_config["embedding"]["performance_path"] = f"stage_{file_ind}.yaml"
        train_config["embedding"]["log_dir"] = f"logging/version_{file_ind}"
        yaml.dump(train_config["embedding"], f)

    with open(os.path.join(FILTER_DIR, f"{train_config_name}_{file_ind}.yaml"), "w") as f:
        train_config["filter"][
            "input_dir"] = f"$CFS/m3443/usr/daniel/dataset/embedding_{file_ind}"
        train_config["filter"][
            "output_dir"] = f"$CFS/m3443/usr/daniel/dataset/filter_{file_ind}"
        train_config["filter"]["performance_path"] = f"stage_{file_ind}.yaml"
        train_config["filter"]["log_dir"] = f"logging/version_{file_ind}"
        yaml.dump(train_config["filter"], f)

    with open(os.path.join(GNN_DIR, f"{train_config_name}_{file_ind}.yaml"), "w") as f:
        train_config["gnn"][
            "input_dir"] = f"$CFS/m3443/usr/daniel/dataset/filter_{file_ind}"
        train_config["gnn"][
            "output_dir"] = f"$CFS/m3443/usr/daniel/dataset/gnn_{file_ind}"
        train_config["gnn"]["performance_path"] = f"stage_{file_ind}.yaml"
        train_config["gnn"]["log_dir"] = f"logging/version_{file_ind}"
        yaml.dump(train_config["gnn"], f)

    with open(f"../analysis/HSF/configs/gnn_{file_ind}.yaml", "r") as f:
        gnn_config = yaml.load(f, yaml.FullLoader)

    with open(f"../analysis/HSF/configs/gnn_{file_ind}.yaml", "w") as f:
        gnn_config['event']['files']['gnn_processed'][
            'file'] = "/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn_" + str(file_ind) + "/test/{evtid:04}"

        yaml.dump(gnn_config, f)
