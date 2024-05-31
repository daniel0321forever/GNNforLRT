import pandas as pd
import yaml
from datetime import datetime


def to_csv(config_paths: list, ind: int, title: str):
    dfs = []

    for config_path in config_paths:
        with open(config_path) as f:
            config: dict = yaml.load(f, yaml.FullLoader)

        config.pop("input_dir", None)
        config.pop("output_dir", None)
        config.pop("project", None)
        config.pop("checkpoint_path", None)
        config["date"] = datetime.today().isoformat()
        config["type"] = ind
        for key in config.keys():

            if isinstance(config[key], list):
                if isinstance(config[key][0], list):
                    config[key] = config[key][0]
                config[key] = ",".join(map(str, config[key]))

        dfs.append(pd.DataFrame(config, index=[0]))

    df = pd.concat(dfs, axis=1)
    df.to_csv(f"{title}.csv", mode="a")


if __name__ == "__main__":
    for i in range(1, 4):
        to_csv([f"./LightningModules/Embedding/noPU_train_{i}.yaml", f"./LightningModules/Filter/noPU_train_{i}.yaml",
                f"./LightningModules/GNN/noPU_train_{i}.yaml"], title="param", ind=i)
