import pandas as pd
import yaml


def to_csv(config_paths: list, title: str):
    dfs = []

    for config_path in config_paths:
        with open(config_path) as f:
            config: dict = yaml.load(f, yaml.FullLoader)

        config.pop("input_dir", None)
        config.pop("output_dir", None)
        config.pop("project", None)
        config.pop("checkpoint_path", None)
        for key in config.keys():

            if isinstance(config[key], list):
                if isinstance(config[key][0], list):
                    config[key] = config[key][0]
                config[key] = ",".join(map(str, config[key]))

        dfs.append(pd.DataFrame(config, index=[0]))

    df = pd.concat(dfs, axis=1)
    df.to_csv(f"{title}.csv")


if __name__ == "__main__":
    to_csv(["./LightningModules/Embedding/noPU_train.yaml", "./LightningModules/Filter/noPU_train.yaml",
           "./LightningModules/GNN/noPU_train.yaml"], title="param")
