'''
Summarize checkpoint.pt files found in each subdirectory of the current directory.
'''

import os
import sys
import torch
import argparse


def summarize_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return None, str(e)

    if not isinstance(ckpt, dict):
        return None, f'Unexpected type: {type(ckpt).__name__}'

    info = {}

    info['epoch_finished'] = ckpt.get('epoch_finished', 'N/A')
    info['best_rmse'] = ckpt.get('best_rmse', 'N/A')

    rmse_train = ckpt.get('rmse_train', [])
    rmse_test = ckpt.get('rmse_test', [])
    info['last_rmse_train'] = rmse_train[-1] if rmse_train else 'N/A'
    info['last_rmse_test'] = rmse_test[-1] if rmse_test else 'N/A'
    info['min_rmse_train'] = min(rmse_train) if rmse_train else 'N/A'
    info['min_rmse_test'] = min(v for v in rmse_test if v >= 0) if any(v >= 0 for v in rmse_test) else 'N/A'

    train_time = ckpt.get('train_time', [])
    val_time = ckpt.get('validation_time', [])
    info['total_train_time_s'] = sum(train_time) if train_time else 'N/A'
    info['total_val_time_s'] = sum(val_time) if val_time else 'N/A'

    return info, None


def fmt(val, decimals=6):
    if isinstance(val, float):
        return f'{val:.{decimals}f}'
    return str(val)


def main():
    parser = argparse.ArgumentParser(description='Summarize checkpoint.pt files in subdirectories')
    parser.add_argument('-d', '--directory', default='.', help='Root directory to search (default: current dir)')
    parser.add_argument('-f', '--filename', default='checkpoint.pt', help='Checkpoint filename (default: checkpoint.pt)')
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    entries = sorted(
        e.name for e in os.scandir(root)
        if e.is_dir()
    )

    if not entries:
        print(f'No subdirectories found in {root}')
        return

    found = []
    for name in entries:
        ckpt_path = os.path.join(root, name, args.filename)
        if os.path.isfile(ckpt_path):
            found.append((name, ckpt_path))

    if not found:
        print(f'No {args.filename} files found in subdirectories of {root}')
        return

    # Header
    col_w = 16
    header = (
        f"{'Directory':<30}  "
        f"{'Epoch':>8}  "
        f"{'BestRMSE':>{col_w}}  "
        f"{'LastTrain':>{col_w}}  "
        f"{'LastTest':>{col_w}}  "
        f"{'MinTrain':>{col_w}}  "
        f"{'MinTest':>{col_w}}  "
        f"{'TrainTime(s)':>14}  "
        f"{'ValTime(s)':>12}"
    )
    print(header)
    print('-' * len(header))

    for name, path in found:
        info, err = summarize_checkpoint(path)
        if err:
            print(f"{name:<30}  ERROR: {err}")
            continue
        print(
            f"{name:<30}  "
            f"{fmt(info['epoch_finished']):>8}  "
            f"{fmt(info['best_rmse']):>{col_w}}  "
            f"{fmt(info['last_rmse_train']):>{col_w}}  "
            f"{fmt(info['last_rmse_test']):>{col_w}}  "
            f"{fmt(info['min_rmse_train']):>{col_w}}  "
            f"{fmt(info['min_rmse_test']):>{col_w}}  "
            f"{fmt(info['total_train_time_s'], 1):>14}  "
            f"{fmt(info['total_val_time_s'], 1):>12}"
        )


if __name__ == '__main__':
    main()
