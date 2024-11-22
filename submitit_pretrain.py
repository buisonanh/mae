import argparse
import os
from pathlib import Path

import main_pretrain as trainer


def parse_args():
    # Use the trainer's argument parser directly
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Kaggle MAE pretrain", parents=[trainer_parser], conflict_handler='resolve')
    parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs to use (Kaggle allows 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure the output directory exists if it's not already defined
    if not hasattr(args, 'output_dir') or not args.output_dir:
        args.output_dir = "./output"
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up single GPU
    args.gpu = 0  # Kaggle uses GPU index 0
    args.rank = 0  # Single process
    args.world_size = 1
    args.dist_url = "env://"  # Distributed training not needed for single GPU

    print(f"Running MAE pretraining on GPU {args.gpu}")
    trainer.main(args)


if __name__ == "__main__":
    main()
