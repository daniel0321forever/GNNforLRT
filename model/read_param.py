import pandas as pd
import yaml
from datetime import datetime
import argparse

"""
Add pipeline index
"""


def to_csv(config_path: str, title: str):
    dfs = []

    with open(config_path) as f:
        full_config: dict = yaml.load(f, yaml.FullLoader)
        for stage in full_config.keys():
            config = full_config[stage]
            config.pop("input_dir", None)
            output_dir = config.pop("output_dir", None)
            config.pop("project", None)
            p_path = config.pop("performance_path", None)
            log_dir = config.pop("log_dir", None)
            config.pop("checkpoint_path", None)
            for key in config.keys():

                if isinstance(config[key], list):
                    if isinstance(config[key][0], list):
                        config[key] = config[key][0]
                    config[key] = ",".join(map(str, config[key]))

            dfs.append(pd.DataFrame(config, index=[0]))

    df = pd.concat(dfs, axis=1)
    df.insert(0, "config", config_path)
    df.insert(2, "stage_performance", p_path)
    df.insert(3, "logging", log_dir)
    df.insert(4, "output", output_dir)
    df.to_csv(f"{title}.csv", mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    to_csv(config_path, title="param")
