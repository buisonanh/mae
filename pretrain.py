# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import uuid
from pathlib import Path
import main_pretrain as trainer


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Direct Training for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to use")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="", type=str, help="Directory to save job outputs. Leave empty for automatic.")

    return parser.parse_args()


def get_shared_folder() -> Path:
    """Get a shared folder for outputs (defaults to Kaggle working directory)."""
    p = Path("/kaggle/working/experiments")
    p.mkdir(exist_ok=True)
    return p


def get_init_file():
    """Create a unique init file for distributed training synchronization."""
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer:
    def __init__(self, args):
        self.args = args

    def start_training(self):
        """Start the training process directly."""
        print("Starting training...")
        self._setup_gpu_args()
        trainer.main(self.args)

    def _setup_gpu_args(self):
        """Set up GPU-related arguments."""
        self.args.gpu = 0  # For single-GPU training on Kaggle
        self.args.rank = 0
        self.args.world_size = 1
        print(f"Training setup: GPU {self.args.gpu}, rank {self.args.rank}, world size {self.args.world_size}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder()

    # Set distributed training configurations (even if running locally)
    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    print("Initializing Trainer...")
    trainer = Trainer(args)
    print("Starting the training process...")
    trainer.start_training()


if __name__ == "__main__":
    main()
