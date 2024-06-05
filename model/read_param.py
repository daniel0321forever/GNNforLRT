import pandas as pd
import yaml
from datetime import datetime
import argparse


def to_csv(config_path: str, title: str):
    dfs = []

    with open(config_path) as f:
        full_config: dict = yaml.load(f, yaml.FullLoader)
        for stage in full_config.keys():
            config = full_config[stage]
            config.pop("input_dir", None)
            config.pop("output_dir", None)
            config.pop("project", None)
            config.pop("performance_path", None)
            config.pop("log_dir", None)
            config.pop("checkpoint_path", None)
            for key in config.keys():

                if isinstance(config[key], list):
                    if isinstance(config[key][0], list):
                        config[key] = config[key][0]
                    config[key] = ",".join(map(str, config[key]))

            dfs.append(pd.DataFrame(config, index=[0]))

    df = pd.concat(dfs, axis=1)
    df.to_csv(f"{title}.csv", mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    to_csv(config_path, title="param")
