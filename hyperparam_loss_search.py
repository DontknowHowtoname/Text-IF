"""
Hyperparameter grid search for max_ratio x ssim_ratio in fusion loss.
Continues training from a given checkpoint and searches for the best
loss ratio combination ranked by val_total_loss.

Usage:
    python hyperparam_loss_search.py \
        --resume experiments/hyperparam_search/20260519-220812/lr0.0001_rw0.05/weights/checkpoint.pth \
        --lr 1e-4 --recon_weight 0.05 \
        --max_ratio_values 2 4 6 8 \
        --ssim_ratio_values 0.5 1 2 4 \
        --epochs 20
"""
import os
import subprocess
import sys
import itertools
import json
import shutil
import glob
import datetime
import argparse


def parse_tb_best_val(log_dir: str) -> dict:
    """Parse TensorBoard log and return best val metrics."""
    from tbparse import SummaryReader
    reader = SummaryReader(log_dir)
    df = reader.scalars

    result = {}
    for tag in df['tag'].unique():
        if not tag.startswith('val_'):
            continue
        sub = df[df['tag'] == tag].sort_values('step')
        vals = sub['value'].values
        steps = sub['step'].values
        idx = vals.argmin()
        result[tag] = {
            'best_value': float(vals[idx]),
            'best_epoch': int(steps[idx]),
            'final_value': float(vals[-1]),
        }
    return result


def run_training(train_script: str, params: dict, output_dir: str, extra_args: list) -> str:
    """Run a single training job and return the experiment directory."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, train_script,
        '--lr', str(params['lr']),
        '--recon_weight', str(params['recon_weight']),
        '--max_ratio', str(params['max_ratio']),
        '--ssim_ratio', str(params['ssim_ratio']),
        '--epochs', str(params['epochs']),
        '--batch-size', str(params['batch_size']),
        '--gpu_id', str(params['gpu_id']),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"Training: max_ratio={params['max_ratio']}, ssim_ratio={params['ssim_ratio']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=os.path.dirname(train_script) or '.')

    if result.returncode != 0:
        print(f"[WARNING] Training failed for max_ratio={params['max_ratio']}, "
              f"ssim_ratio={params['ssim_ratio']}")
        return None

    # Find the latest experiment directory matching the pattern
    pattern = os.path.join(os.path.dirname(train_script) or '.',
                           'experiments', 'TextIF_full_recon_v2_ft_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not dirs:
        return None

    exp_dir = dirs[-1]
    # Move to organized output directory
    run_name = f"mr{params['max_ratio']}_sr{params['ssim_ratio']}"
    dest = os.path.join(output_dir, run_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(exp_dir, dest)

    return dest


def main():
    parser = argparse.ArgumentParser(
        description='Grid search over max_ratio x ssim_ratio for fusion loss')
    parser.add_argument('--train_script', type=str,
                        default='train_fusion_full_recon_v2_ft.py',
                        help='Training script path')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/loss_search',
                        help='Directory to store all search results')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs per run (default: 20)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU device id')

    # Fixed training hyperparams (from previous best run)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (fixed, from previous best)')
    parser.add_argument('--recon_weight', type=float, default=0.05,
                        help='Reconstruction weight (fixed, from previous best)')

    # Grid search spaces
    parser.add_argument('--max_ratio_values', type=float, nargs='+',
                        default=[2, 4, 6, 8],
                        help='max_ratio candidates (default: 2 4 6 8)')
    parser.add_argument('--ssim_ratio_values', type=float, nargs='+',
                        default=[0.5, 1, 2, 4],
                        help='ssim_ratio candidates (default: 0.5 1 2 4)')

    # Checkpoint to continue from
    parser.add_argument('--resume', type=str, required=True,
                        help='Checkpoint path to continue training from')

    # Pass-through arguments
    parser.add_argument('--low_light_path', type=str, default=None)
    parser.add_argument('--over_exposure_path', type=str, default=None)
    parser.add_argument('--ir_low_contrast_path', type=str, default=None)
    parser.add_argument('--ir_noise_path', type=str, default=None)

    args = parser.parse_args()

    # Build parameter grid
    mr_list = args.max_ratio_values
    sr_list = args.ssim_ratio_values
    grid = list(itertools.product(mr_list, sr_list))
    total = len(grid)

    print(f"Loss parameter grid search: {total} combinations")
    print(f"  max_ratio:   {mr_list}")
    print(f"  ssim_ratio:  {sr_list}")
    print(f"  Fixed lr:    {args.lr}")
    print(f"  Fixed rw:    {args.recon_weight}")
    print(f"  Resume from: {args.resume}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Output:      {args.output_dir}")
    print()

    # Build extra args for training script
    extra_args = ['--resume', args.resume]
    if args.low_light_path:
        extra_args += ['--low_light_path', args.low_light_path]
    if args.over_exposure_path:
        extra_args += ['--over_exposure_path', args.over_exposure_path]
    if args.ir_low_contrast_path:
        extra_args += ['--ir_low_contrast_path', args.ir_low_contrast_path]
    if args.ir_noise_path:
        extra_args += ['--ir_noise_path', args.ir_noise_path]

    results = []
    output_dir = os.path.join(args.output_dir,
                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    for i, (mr, sr) in enumerate(grid):
        print(f"\n[{i+1}/{total}] max_ratio={mr}, ssim_ratio={sr}")

        params = {
            'lr': args.lr,
            'recon_weight': args.recon_weight,
            'max_ratio': mr,
            'ssim_ratio': sr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'gpu_id': args.gpu_id,
        }

        exp_dir = run_training(args.train_script, params, output_dir, extra_args)

        entry = {'max_ratio': mr, 'ssim_ratio': sr, 'exp_dir': None}

        if exp_dir and os.path.isdir(os.path.join(exp_dir, 'log')):
            try:
                val_metrics = parse_tb_best_val(os.path.join(exp_dir, 'log'))
                entry['exp_dir'] = exp_dir
                entry['metrics'] = val_metrics
                best_val = val_metrics.get('val_total_loss', {})
                print(f"  Best val_total_loss: {best_val.get('best_value', 'N/A'):.4f} "
                      f"@ epoch {best_val.get('best_epoch', 'N/A')}")
            except Exception as e:
                print(f"  [ERROR] Failed to parse logs: {e}")
                entry['error'] = str(e)
        else:
            print(f"  [ERROR] No experiment directory or log found")
            entry['error'] = 'no_log_dir'

        results.append(entry)

        # Save intermediate results after each run
        with open(os.path.join(output_dir, 'search_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary
    valid = [r for r in results if 'metrics' in r]
    if valid:
        valid.sort(key=lambda x: x['metrics'].get('val_total_loss', {}).get('best_value', float('inf')))

        print("\n" + "=" * 90)
        print("SEARCH RESULTS (ranked by best val_total_loss)")
        print("=" * 90)
        print(f"{'Rank':<5} {'MaxR':<8} {'SSIMR':<8} {'Best Val Loss':<15} "
              f"{'Best Epoch':<12} {'Final Val Loss':<15}")
        print("-" * 90)

        for rank, r in enumerate(valid, 1):
            m = r['metrics'].get('val_total_loss', {})
            print(f"{rank:<5} {r['max_ratio']:<8} {r['ssim_ratio']:<8} "
                  f"{m.get('best_value', 'N/A'):<15.4f} "
                  f"{m.get('best_epoch', 'N/A'):<12} "
                  f"{m.get('final_value', 'N/A'):<15.4f}")

        best = valid[0]
        print(f"\nBest: max_ratio={best['max_ratio']}, ssim_ratio={best['ssim_ratio']}, "
              f"val_total_loss={best['metrics']['val_total_loss']['best_value']:.4f}")
    else:
        print("\n[ERROR] No valid results found.")

    print(f"\nFull results saved to: {os.path.join(output_dir, 'search_results.json')}")


if __name__ == '__main__':
    main()
