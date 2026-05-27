"""
Hyperparameter grid search for Text_IF_Recon v2 fine-tuning.
Searches over lr x recon_weight combinations, runs training, parses TensorBoard logs,
and reports the best configuration ranked by val_total_loss.
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
        '--epochs', str(params['epochs']),
        '--batch-size', str(params['batch_size']),
        '--gpu_id', str(params['gpu_id']),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"Training: lr={params['lr']}, recon_weight={params['recon_weight']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=os.path.dirname(train_script) or '.')

    if result.returncode != 0:
        print(f"[WARNING] Training failed for lr={params['lr']}, recon_weight={params['recon_weight']}")
        return None

    # Find the latest experiment directory matching the pattern
    pattern = os.path.join(os.path.dirname(train_script) or '.',
                           'experiments', 'TextIF_full_recon_v2_ft_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not dirs:
        return None

    exp_dir = dirs[-1]
    # Move to organized output directory
    run_name = f"lr{params['lr']}_rw{params['recon_weight']}"
    dest = os.path.join(output_dir, run_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(exp_dir, dest)

    return dest


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument('--train_script', type=str,
                        default='train_fusion_full_recon_v2_ft.py',
                        help='Training script path')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/hyperparam_search',
                        help='Directory to store all search results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per run')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU device id')
    # Grid search spaces
    parser.add_argument('--lr_values', type=float, nargs='+',
                        default=[1e-5, 2e-5, 5e-5, 1e-4],
                        help='Learning rate candidates')
    parser.add_argument('--recon_weight_values', type=float, nargs='+',
                        default=[0.01, 0.03, 0.05, 0.1, 0.2],
                        help='Reconstruction weight candidates')

    # Pass-through arguments to training script
    parser.add_argument('--weights', type=str, default='',
                        help='Pretrained weights path (passed to training script)')
    parser.add_argument('--low_light_path', type=str, default=None)
    parser.add_argument('--over_exposure_path', type=str, default=None)
    parser.add_argument('--ir_low_contrast_path', type=str, default=None)
    parser.add_argument('--ir_noise_path', type=str, default=None)

    args = parser.parse_args()

    # Build parameter grid
    lr_list = args.lr_values
    rw_list = args.recon_weight_values
    grid = list(itertools.product(lr_list, rw_list))
    total = len(grid)

    print(f"Grid search: {total} combinations")
    print(f"  lr: {lr_list}")
    print(f"  recon_weight: {rw_list}")
    print(f"  epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    print()

    # Build extra args for training script
    extra_args = []
    if args.weights:
        extra_args += ['--weights', args.weights]
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

    for i, (lr, rw) in enumerate(grid):
        print(f"\n[{i+1}/{total}] lr={lr}, recon_weight={rw}")

        params = {
            'lr': lr,
            'recon_weight': rw,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'gpu_id': args.gpu_id,
        }

        exp_dir = run_training(args.train_script, params, output_dir, extra_args)

        entry = {'lr': lr, 'recon_weight': rw, 'exp_dir': None}

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

        print("\n" + "=" * 80)
        print("SEARCH RESULTS (ranked by best val_total_loss)")
        print("=" * 80)
        print(f"{'Rank':<5} {'LR':<12} {'ReconW':<10} {'Best Val Loss':<15} {'Best Epoch':<12} {'Final Val Loss':<15}")
        print("-" * 80)

        for rank, r in enumerate(valid, 1):
            m = r['metrics'].get('val_total_loss', {})
            print(f"{rank:<5} {r['lr']:<12} {r['recon_weight']:<10} "
                  f"{m.get('best_value', 'N/A'):<15.4f} "
                  f"{m.get('best_epoch', 'N/A'):<12} "
                  f"{m.get('final_value', 'N/A'):<15.4f}")

        best = valid[0]
        print(f"\nBest: lr={best['lr']}, recon_weight={best['recon_weight']}, "
              f"val_total_loss={best['metrics']['val_total_loss']['best_value']:.4f}")
    else:
        print("\n[ERROR] No valid results found.")

    print(f"\nFull results saved to: {os.path.join(output_dir, 'search_results.json')}")


if __name__ == '__main__':
    main()
