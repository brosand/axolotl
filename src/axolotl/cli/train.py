"""
CLI to run training on a model
"""
import logging
from pathlib import Path

import fire
import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
import mlflow

LOG = logging.getLogger("axolotl.cli.train")


def do_cli(config: Path = Path("examples/"), **kwargs):
    with mlflow.start_run():
        # Log parameters (key-value pairs)
        mlflow.log_param("param1", "value1")
        mlflow.log_param("param2", "value2")

        # Log metrics; these can be updated throughout the run
        mlflow.log_metric("foo", 1)
        mlflow.log_metric("foo", 2)
        mlflow.log_metric("foo", 3)
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
