"""
Generate Models CSV from Experiment Directories.

Scans all experiment directories containing 'wm*' suffix, evaluates
each available model (original, hybrid, tflite) on the test set,
and produces a comprehensive CSV with metrics.

Author: Daniele Faggi
Date: February 2026

Usage:
    # Scan all experiments
    python scripts/generate_models_csv.py

    # Debug mode: limit to 2 experiments
    python scripts/generate_models_csv.py --limit 2

    # Custom output CSV
    python scripts/generate_models_csv.py --output my_results.csv

    # Custom data directory
    python scripts/generate_models_csv.py --data_dir data/processed/vww_subset/test
"""

import sys
import os
import re
import time
import argparse
import numpy as np
from datetime import datetime

# Mock matplotlib before any imports that might pull it in
from unittest.mock import MagicMock
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, '.')


# ─────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Generate Models CSV from experiment directories',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--limit', type=int, default=None,
                    help='Limit number of experiments to process (useful for debug)')
parser.add_argument('--output', type=str, default='models_generated.csv',
                    help='Output CSV filename')
parser.add_argument('--data_dir', type=str, default='data/processed/vww_subset/test',
                    help='Path to test dataset directory')
parser.add_argument('--exp_dir', type=str, default='experiment_data',
                    help='Path to experiment data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='Image size for evaluation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for torch model evaluation')
parser.add_argument('--skip_tflite', action='store_true',
                    help='Skip TFLite model evaluation')
parser.add_argument('--skip_hybrid', action='store_true',
                    help='Skip hybrid model evaluation')
parser.add_argument('--skip_original', action='store_true',
                    help='Skip original model evaluation')
parser.add_argument('--disable_xnnpack', action='store_true',
                    help='Disable XNNPACK delegate for TFLite inference '
                         '(use for baseline CPU-only latency without SIMD acceleration)')
args = parser.parse_args()


# ─────────────────────────────────────────────
# Model ID Builder
# ─────────────────────────────────────────────
def build_model_id(exp_name: str, model_type_char: str) -> str:
    """
    Build model ID string from experiment directory name.

    Example:
        kan_16_4_grid5_deg3_..._wm0.75 -> 'kan {type} 16-4 WM 0.75'
        kan_32_24-16_grid5_..._wm1.0   -> 'kan {type} 32-24-16 WM 1.0'

    model_type_char: 'o' (original), 'h' (hybrid), 't' (tflite)
    """
    # Extract dims before 'grid5'
    # Pattern: kan_<DIMS>_grid5_..._wm<WM>
    match = re.match(r'kan_(.+?)_grid5_.*_wm([\d.]+)$', exp_name)
    if not match:
        return f'kan {model_type_char} {exp_name}'

    dims_raw = match.group(1)   # e.g. "16_4" or "32_24-16"
    wm = match.group(2)         # e.g. "0.75"

    # Convert dims: "16_4" -> "16-4", "32_24-16" -> "32-24-16"
    dims = dims_raw.replace('_', '-')

    return f'kan {model_type_char} {dims} WM {wm}'


def get_wm_from_exp(exp_name: str) -> float:
    """Extract width_mult from experiment directory name."""
    match = re.search(r'_wm([\d.]+)$', exp_name)
    if match:
        return float(match.group(1))
    return 1.0


# ─────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────
def get_test_loader(data_dir: str, img_size: int, batch_size: int):
    """Create test DataLoader with standard ImageNet normalization."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, dataset.classes


# ─────────────────────────────────────────────
# Torch Model Loader
# ─────────────────────────────────────────────
def load_torch_model(model_path: Path):
    """Load a PyTorch checkpoint and return (model, size_mb, total_params, trainable_params).
    
    NOTE: size_mb is computed from model parameters (weights only),
    NOT from file size (which includes optimizer state in 'best' checkpoints).
    """
    from models.kan_model import KANImageClassifier

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Read config from checkpoint
    saved_config = None
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
        elif 'model_info' in checkpoint and 'config' in checkpoint['model_info']:
            saved_config = checkpoint['model_info']['config']

    if saved_config and 'preprocessor' in saved_config:
        preproc = saved_config['preprocessor']
        kan = saved_config['kan']
        feature_dim = preproc['output_features']
        hidden_dims = kan.get('hidden_dims', [feature_dim // 2])
        grid = kan.get('grid', 5)
        degree = kan.get('degree', 3)
        preprocessor_type = preproc.get('preprocessor_type', 'mobilenetv3_small')
        width_mult = preproc.get('width_mult', 1.0)
    else:
        # Fallback - read wm from filename parent name
        wm_str = re.search(r'_wm([\d.]+)', str(model_path))
        width_mult = float(wm_str.group(1)) if wm_str else 1.0
        feature_dim = 32
        hidden_dims = [16]
        grid = 5
        degree = 3
        preprocessor_type = 'mobilenetv3_small'

    model = KANImageClassifier(
        input_channels=3,
        img_size=224,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        preprocessor_type=preprocessor_type,
        width_mult=width_mult,
        preprocessor_pretrained=False,
    )

    # Load weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'quantized_state' in checkpoint:
            model.load_state_dict(checkpoint['quantized_state'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.cpu()  # force CPU — all benchmarks must run on the same device

    # Param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size from model weights only (not file size, which may include optimizer state)
    # Same method as analyze.py: sum of parameter bytes
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_bytes + buf_bytes) / (1024 * 1024)

    return model, size_mb, total_params, trainable_params


# ─────────────────────────────────────────────
# Hybrid Model Loader
# ─────────────────────────────────────────────
def load_hybrid_model(model_path: Path):
    """Load a hybrid INT8 checkpoint (saved by quantize_hybrid_manual.py).

    The checkpoint stores:
      - quantized_state: dict with FP32 weights for KAN layers and INT8 for MobileNet
      - scales:          dict mapping 'name_scale' -> scale tensor for dequantization
      - config:          model configuration dict

    We dequantize INT8 -> FP32 before loading into the model so inference works normally.
    size_mb is computed from the quantized file size (reflects actual storage cost).
    """
    from models.kan_model import KANImageClassifier

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict) or 'quantized_state' not in checkpoint:
        raise ValueError(f"Not a hybrid checkpoint: {model_path.name}. "
                         f"Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

    quantized_state = checkpoint['quantized_state']
    scales = checkpoint.get('scales', {})

    # Build model from saved config
    saved_config = checkpoint.get('config', {})
    if saved_config and 'preprocessor' in saved_config:
        preproc = saved_config['preprocessor']
        kan = saved_config['kan']
        feature_dim = preproc['output_features']
        hidden_dims = kan.get('hidden_dims', [feature_dim // 2])
        grid = kan.get('grid', 5)
        degree = kan.get('degree', 3)
        preprocessor_type = preproc.get('preprocessor_type', 'mobilenetv3_small')
        width_mult = preproc.get('width_mult', 1.0)
    else:
        wm_str = re.search(r'_wm([\d.]+)', str(model_path))
        width_mult = float(wm_str.group(1)) if wm_str else 1.0
        feature_dim = 32
        hidden_dims = [16]
        grid = 5
        degree = 3
        preprocessor_type = 'mobilenetv3_small'

    model = KANImageClassifier(
        input_channels=3,
        img_size=224,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        preprocessor_type=preprocessor_type,
        width_mult=width_mult,
        preprocessor_pretrained=False,
    )

    # Dequantize INT8 weights back to FP32 before loading
    # (same logic as quantize_hybrid_manual.py Step 2)
    fp32_state = {}
    for name, tensor in quantized_state.items():
        if name.endswith('_scale'):
            continue  # skip scale entries
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int8:
            scale_key = name + '_scale'
            scale = scales.get(scale_key, quantized_state.get(scale_key))
            if scale is not None:
                fp32_state[name] = tensor.float() * scale.item()
            else:
                fp32_state[name] = tensor.float()  # fallback: just cast
        else:
            fp32_state[name] = tensor

    model.load_state_dict(fp32_state, strict=True)
    model.eval()
    model.cpu()  # force CPU — all benchmarks must run on the same device

    # Param counts (after dequantization model is full FP32)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0  # hybrid models are not trainable

    # Size: use file size (reflects actual quantized storage, not FP32 equivalent)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    return model, size_mb, total_params, trainable_params



# ─────────────────────────────────────────────
# Peak Activation Memory Estimator (PyTorch)
# ─────────────────────────────────────────────
def estimate_peak_activation_kb(model: torch.nn.Module, img_size: int = 224) -> float:
    """Estimate peak activation memory (KB) for a single-image forward pass.

    NOTE: This is an UPPER BOUND. It sums ALL intermediate output tensors
    across every submodule, without accounting for memory reuse between
    non-concurrent activations. The real runtime peak is always smaller.
    Still useful as a consistent, comparable metric across models.
    """
    total_bytes = [0]

    def _hook(_module, _inp, output):
        if isinstance(output, torch.Tensor):
            total_bytes[0] += output.numel() * output.element_size()

    handles = [
        m.register_forward_hook(_hook)
        for m in model.modules()
        if m is not model   # skip the root module (would double-count)
    ]
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            model(dummy)
    finally:
        for h in handles:
            h.remove()

    return total_bytes[0] / 1024.0


# ─────────────────────────────────────────────
# Torch Evaluation
# ─────────────────────────────────────────────
def evaluate_torch(model, loader, classes, desc='Evaluating', n_latency_samples=200):
    """Return (accuracy, avg_latency_ms, precision, recall, f1, cm) for a torch model.

    Fairness note:
      - Accuracy/metrics are computed on BATCHES (fast).
      - Latency is measured on SINGLE images (batch_size=1), same as TFLite.
        This reflects real IoT deployment where images arrive one at a time.
      - n_latency_samples: how many single-image inferences to time.
    """
    all_preds = []
    all_targets = []

    # ── Phase 1: accuracy on batches (fast) ───────────────────────
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"    {desc} [acc]", unit='batch',
                    leave=False, dynamic_ncols=True)
        for inputs, targets in pbar:
            inputs = inputs.cpu()
            targets = targets.cpu()
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
            cur_acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
            pbar.set_postfix(acc=f'{cur_acc:.1f}%')

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = 100 * np.mean(all_preds == all_targets)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    # ── Phase 2: latency on single images (fair vs TFLite) ─────────
    latencies = []
    collected = 0
    with torch.no_grad():
        pbar2 = tqdm(total=n_latency_samples, desc=f"    {desc} [lat]",
                     unit='img', leave=False, dynamic_ncols=True)
        for inputs, _ in loader:
            for i in range(inputs.size(0)):
                if collected >= n_latency_samples:
                    break
                img = inputs[i:i+1].cpu()   # batch of 1
                t0 = time.time()
                model(img)
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)
                collected += 1
                pbar2.update(1)
                pbar2.set_postfix(lat=f'{latencies[-1]:.1f}ms')
            if collected >= n_latency_samples:
                break
        pbar2.close()

    avg_lat = float(np.mean(latencies))

    return accuracy, avg_lat, prec * 100, rec * 100, f1 * 100, cm



# ─────────────────────────────────────────────
# TFLite Evaluation
# ─────────────────────────────────────────────
def evaluate_tflite(model_path: Path, loader, classes, use_xnnpack: bool = True):
    """Return (accuracy, avg_latency_ms, precision, recall, f1, cm, size_mb, clean_params, total_buffers) for TFLite.

    use_xnnpack: if False, disables XNNPACK delegate so latency reflects
                 plain CPU scalar/vector ops without SIMD acceleration.
                 This can be useful for fair comparison vs PyTorch on the
                 same CPU ISA level.
    """
    try:
        import tflite_runtime.interpreter as tflite
        _Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        _Interpreter = tf.lite.Interpreter

    # Build interpreter with optional XNNPACK control
    # (mirrors the pattern in test_tflite.py)
    interpreter = None
    if use_xnnpack:
        # Default: let TFLite decide (usually enables XNNPACK on supported platforms)
        try:
            interpreter = _Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"     [WARN] Standard TFLite init failed: {e} -- retrying without delegates")
            interpreter = None

    if interpreter is None:
        # Disable XNNPACK: set env var + explicit empty delegates
        os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                experimental_delegates=[],
                num_threads=1,
            )
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
                interpreter = LiteRTInterpreter(model_path=str(model_path))
            except ImportError:
                interpreter = _Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_shape = input_details[0]['shape']

    # ── Count parameters from weight tensors ──────────────────────
    # HACK: The TFLite Python API doesn't expose a direct "is_constant" flag per
    # tensor, so we use a two-invocation trick to distinguish weight tensors from
    # runtime activation buffers:
    #   - Invoke once with all-zeros input  → snapshot tensor bytes
    #   - Invoke again with all-ones input  → re-read tensor bytes
    #   - Tensors whose bytes are IDENTICAL across both calls = constant weights
    #   - Tensors that CHANGED = activation maps (depend on input) → excluded
    # This correctly handles bias=0 (included, since bytes are stable) and avoids
    # inflating the count with large intermediate activation maps (e.g. 112×112×16).
    #
    # total_buffers = all internal non-io tensors (kept for diagnostics)
    # clean_params  = only constant weight tensors (comparable to PyTorch params)

    io_indices = {d['index'] for d in input_details + output_details}

    def _make_dummy(val):
        d = np.full(input_shape, val, dtype=np.float32)
        if input_dtype == np.uint8:
            d = np.clip(d * 255, 0, 255).astype(np.uint8)
        elif input_dtype == np.int8:
            d = np.clip(d * 127, -128, 127).astype(np.int8)
        return d

    # First pass: invoke with zeros
    interpreter.set_tensor(input_index, _make_dummy(0.0))
    interpreter.invoke()
    state0 = {}
    for t in interpreter.get_tensor_details():
        if t['index'] in io_indices:
            continue
        shape = t['shape']
        if len(shape) == 0 or not all(s > 0 for s in shape):
            continue
        try:
            state0[t['index']] = interpreter.get_tensor(t['index']).tobytes()
        except Exception:
            pass

    # Second pass: invoke with ones
    interpreter.set_tensor(input_index, _make_dummy(1.0))
    interpreter.invoke()

    clean_params = 0
    total_buffers = 0
    for t in interpreter.get_tensor_details():
        if t['index'] in io_indices:
            continue
        shape = t['shape']
        if len(shape) == 0 or not all(s > 0 for s in shape):
            continue
        n = int(np.prod(shape))
        total_buffers += n
        # Constant weight: bytes identical across both invocations
        if t['index'] in state0:
            try:
                state1 = interpreter.get_tensor(t['index']).tobytes()
                if state0[t['index']] == state1:
                    clean_params += n
            except Exception:
                pass

    all_preds = []
    all_targets = []
    latencies = []
    total_samples = len(loader.dataset)

    pbar = tqdm(total=total_samples, desc='    Inferring',
                unit='img', leave=False, dynamic_ncols=True)

    for images, labels in loader:
        for j in range(len(images)):
            image = images[j]
            label = labels[j].item()

            # NHWC vs NCHW
            if input_shape[3] == 3 and input_shape[1] != 3:
                input_data = image.permute(1, 2, 0).numpy()
                input_data = np.expand_dims(input_data, axis=0)
            else:
                input_data = image.numpy()
                input_data = np.expand_dims(input_data, axis=0)

            # Quantization
            if input_dtype == np.uint8:
                scale, zp = input_details[0]['quantization']
                input_data = (input_data / scale + zp).astype(np.uint8)
            elif input_dtype == np.int8:
                scale, zp = input_details[0]['quantization']
                input_data = (input_data / scale + zp).astype(np.int8)
            else:
                input_data = input_data.astype(np.float32)

            t0 = time.time()
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            t1 = time.time()

            latencies.append((t1 - t0) * 1000)
            prediction = int(np.argmax(output_data))

            all_preds.append(prediction)
            all_targets.append(label)

            # Update progress bar every sample
            cur_acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
            pbar.update(1)
            pbar.set_postfix(acc=f'{cur_acc:.1f}%', lat=f'{latencies[-1]:.1f}ms')

    pbar.close()

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = 100 * np.mean(all_preds == all_targets)
    avg_lat = np.mean(latencies)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    return accuracy, avg_lat, prec * 100, rec * 100, f1 * 100, cm, size_mb, clean_params, total_buffers



# ─────────────────────────────────────────────
# Determine dtype label
# ─────────────────────────────────────────────
def get_dtype_label(model_path: Path) -> str:
    name = model_path.name
    if 'hybrid' in name:
        return 'int8+float32'
    if name.endswith('.tflite'):
        return 'int8'
    return 'float32'


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    exp_root = Path(args.exp_dir)
    data_dir = args.data_dir
    img_size = args.img_size

    if not exp_root.exists():
        print(f"[ERROR] Experiment directory not found: {exp_root}")
        sys.exit(1)

    if not Path(data_dir).exists():
        print(f"[ERROR] Test data directory not found: {data_dir}")
        sys.exit(1)

    # ── Scan experiments ──────────────────────────
    # Only dirs that end with _wm<float>
    wm_pattern = re.compile(r'_wm[\d.]+$')
    all_exps = sorted([
        d for d in exp_root.iterdir()
        if d.is_dir() and wm_pattern.search(d.name)
    ])

    if args.limit:
        all_exps = all_exps[:args.limit]

    print(f"\n{'='*70}")
    print(f"  Model Benchmark CSV Generator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        print(f"  [INFO] CUDA available but IGNORED -- all inference runs on CPU")
        print(f"         (ensures fair latency comparison across model types)")
    else:
        print(f"  [INFO] Running on CPU")
    xnnpack_status = 'DISABLED' if args.disable_xnnpack else 'ENABLED'
    print(f"  [INFO] TFLite XNNPACK: {xnnpack_status}")
    print(f"{'='*70}")
    print(f"  Experiments found : {len(all_exps)}")
    print(f"  Data dir          : {data_dir}")
    print(f"  Output CSV        : {args.output}")
    print(f"  Limit             : {args.limit or 'all'}")
    skip_flags = []
    if args.skip_original: skip_flags.append('original')
    if args.skip_hybrid:   skip_flags.append('hybrid')
    if args.skip_tflite:   skip_flags.append('tflite')
    if skip_flags:
        print(f"  Skipping          : {', '.join(skip_flags)}")
    print(f"{'='*70}\n")

    # ── Test loader ───────────────────────────────
    print("Loading test dataset...")
    loader, classes = get_test_loader(data_dir, img_size, args.batch_size)
    print(f"  Classes     : {classes}")
    print(f"  Samples     : {len(loader.dataset)}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Batches     : {len(loader)}")
    print()

    rows = []
    t_start_all = time.time()
    ok_count = 0
    broken_count = 0
    skip_count = 0

    for exp_idx, exp_dir in enumerate(all_exps):
        exp_name = exp_dir.name
        models_dir = exp_dir / 'models'

        # -- Experiment header --------------------------
        print(f"\n+{'-'*68}+")
        print(f"| [{exp_idx+1:>3}/{len(all_exps)}] {exp_name[:60]:<60} |")
        print(f"+{'-'*68}+")

        if not models_dir.exists():
            print("  [WARN] No models/ directory -- skipping.")
            skip_count += 1
            continue

        # ── Helper to build a row dict ───────────
        def make_row(model_id, model_type, dtype,
                     size_mb, total_params, trainable_params, total_params_and_buffers,
                     accuracy, latency_ms,
                     precision, recall, f1, cm,
                     peak_activation_kb=None):
            return {
                'Model_ID': model_id,
                'type': model_type,
                'size_MB': round(size_mb, 3),
                'dtype': dtype,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'total_params_and_buffers': total_params_and_buffers,
                'peak_activation_kb': round(peak_activation_kb, 1) if peak_activation_kb is not None else None,
                'accuracy_%': round(accuracy, 2),
                'latency_single_ms': round(latency_ms, 2),
                'precision_%': round(precision, 2),
                'recall_%': round(recall, 2),
                'f1_%': round(f1, 2),
                'cm_tn': int(cm[0, 0]),
                'cm_fp': int(cm[0, 1]),
                'cm_fn': int(cm[1, 0]),
                'cm_tp': int(cm[1, 1]),
            }

        def make_broken_row(model_id, model_type, dtype, model_path):
            """Row inserted when a model exists but fails to load/evaluate."""
            size_mb = model_path.stat().st_size / (1024 * 1024)
            return {
                'Model_ID': model_id,
                'type': model_type,
                'size_MB': round(size_mb, 3),
                'dtype': dtype,
                'total_params': None,
                'trainable_params': None,
                'total_params_and_buffers': None,
                'peak_activation_kb': None,
                'accuracy_%': None,
                'latency_single_ms': None,
                'precision_%': None,
                'recall_%': None,
                'f1_%': None,
                'cm_tn': None,
                'cm_fp': None,
                'cm_fn': None,
                'cm_tp': None,
            }

        # ── ORIGINAL model ────────────────────────
        # Use 'best' checkpoint (best val accuracy, same source used for TFLite/hybrid export).
        # size_MB is computed from model parameters, not file size (which includes optimizer state).
        orig_path = models_dir / 'kan_person_detector_best.pt'
        if not args.skip_original and orig_path.exists():
            print(f"  >> [original] Loading {orig_path.name}  "
                  f"({orig_path.stat().st_size/(1024*1024):.1f} MB on disk)...")
            t0 = time.time()
            try:
                model, size_mb, total_p, train_p = load_torch_model(orig_path)
                peak_kb = estimate_peak_activation_kb(model, img_size)
                print(f"     Loaded in {time.time()-t0:.1f}s | "
                      f"Params={total_p:,} | Size(weights)={size_mb:.3f} MB | "
                      f"PeakAct={peak_kb:.0f} KB")
                acc, lat, prec, rec, f1, cm = evaluate_torch(
                    model, loader, classes,
                    desc=f'original [{exp_idx+1}/{len(all_exps)}]')
                model_id = build_model_id(exp_name, 'o')
                rows.append(make_row(model_id, 'original', 'float32',
                                     size_mb, total_p, train_p, total_p,
                                     acc, lat, prec, rec, f1, cm,
                                     peak_activation_kb=peak_kb))
                print(f"  OK [original] Acc={acc:.2f}%  Lat={lat:.2f}ms  "
                      f"Prec={prec:.2f}%  Rec={rec:.2f}%  F1={f1:.2f}%")
                ok_count += 1
            except Exception as e:
                print(f"  !! [original] ERROR: {e}")
                print(f"               -> Adding broken row (metrics=NaN)")
                model_id = build_model_id(exp_name, 'o')
                rows.append(make_broken_row(model_id, 'original', 'float32', orig_path))
                broken_count += 1
        elif not args.skip_original:
            print(f"  -- [original] Not found: {orig_path.name}")

        # -- HYBRID model --------------------------
        hybrid_path = models_dir / 'kan_hybrid.pt'
        if not args.skip_hybrid and hybrid_path.exists():
            print(f"  >> [hybrid]   Loading {hybrid_path.name}  "
                  f"({hybrid_path.stat().st_size/(1024*1024):.1f} MB on disk)...")
            t0 = time.time()
            try:
                model, size_mb, total_p, train_p = load_hybrid_model(hybrid_path)
                peak_kb = estimate_peak_activation_kb(model, img_size)
                print(f"     Loaded in {time.time()-t0:.1f}s | "
                      f"Params={total_p:,} | Size(quantized file)={size_mb:.3f} MB | "
                      f"PeakAct={peak_kb:.0f} KB")
                acc, lat, prec, rec, f1, cm = evaluate_torch(
                    model, loader, classes,
                    desc=f'hybrid   [{exp_idx+1}/{len(all_exps)}]')
                model_id = build_model_id(exp_name, 'h')
                rows.append(make_row(model_id, 'hybrid', 'int8+float32',
                                     size_mb, total_p, train_p, total_p,
                                     acc, lat, prec, rec, f1, cm,
                                     peak_activation_kb=peak_kb))
                print(f"  OK [hybrid]   Acc={acc:.2f}%  Lat={lat:.2f}ms  "
                      f"Prec={prec:.2f}%  Rec={rec:.2f}%  F1={f1:.2f}%")
                ok_count += 1
            except Exception as e:
                print(f"  !! [hybrid]   ERROR: {e}")
                print(f"               -> Adding broken row (metrics=NaN)")
                model_id = build_model_id(exp_name, 'h')
                rows.append(make_broken_row(model_id, 'hybrid', 'int8+float32', hybrid_path))
                broken_count += 1
        elif not args.skip_hybrid:
            print(f"  -- [hybrid]   Not found: {hybrid_path.name}")


        # ── TFLITE model ──────────────────────────
        tflite_path = models_dir / 'model.tflite'
        if not args.skip_tflite and tflite_path.exists():
            print(f"  >> [tflite]   Loading {tflite_path.name}  "
                  f"({tflite_path.stat().st_size/(1024*1024):.2f} MB on disk)...")
            t0 = time.time()
            try:
                acc, lat, prec, rec, f1, cm, size_mb, clean_p, buf_p = evaluate_tflite(
                    tflite_path, loader, classes,
                    use_xnnpack=not args.disable_xnnpack)
                # Peak activation KB estimate for TFLite:
                # activation elements = total_buffers - clean_params (= all non-weight tensors)
                # bytes: TFLite INT8 models use 1 byte/element for activations
                tflite_peak_kb = max(0, buf_p - clean_p) / 1024.0
                model_id = build_model_id(exp_name, 't')
                rows.append(make_row(model_id, 'tflite', 'int8',
                                     size_mb, clean_p, 0, buf_p,
                                     acc, lat, prec, rec, f1, cm,
                                     peak_activation_kb=tflite_peak_kb))
                print(f"  OK [tflite]   Acc={acc:.2f}%  Lat={lat:.2f}ms  "
                      f"Params={clean_p:,} (buffers={buf_p:,})  "
                      f"PeakAct~={tflite_peak_kb:.0f} KB  "
                      f"[{time.time()-t0:.0f}s]")
                ok_count += 1
            except Exception as e:
                print(f"  !! [tflite]   ERROR: {e}")
                print(f"               -> Adding broken row (metrics=NaN)")
                model_id = build_model_id(exp_name, 't')
                rows.append(make_broken_row(model_id, 'tflite', 'int8', tflite_path))
                broken_count += 1
        elif not args.skip_tflite:
            print(f"  -- [tflite]   Not found: {tflite_path.name}")

    # ── Save CSV ──────────────────────────────────
    elapsed = time.time() - t_start_all
    print(f"\n{'='*70}")
    print(f"  Run Summary")
    print(f"{'-'*70}")
    print(f"  Experiments processed : {len(all_exps)}")
    print(f"  Models evaluated OK   : {ok_count}")
    print(f"  Broken (metrics=NaN)  : {broken_count}")
    print(f"  Skipped (no models/)  : {skip_count}")
    print(f"  Total rows            : {len(rows)}")
    print(f"  Elapsed               : {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"{'-'*70}")

    if not rows:
        print("  [WARN] No results collected. CSV not generated.")
        print(f"{'='*70}")
        return

    df = pd.DataFrame(rows)

    # Reorder columns to match models.csv
    col_order = [
        'Model_ID', 'type', 'size_MB', 'dtype',
        'total_params', 'trainable_params', 'total_params_and_buffers',
        'peak_activation_kb',
        'accuracy_%', 'latency_single_ms',
        'precision_%', 'recall_%', 'f1_%',
        'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp',
    ]
    df = df[col_order]

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(f"  [DONE] CSV written to: {output_path.resolve()}")
    print(f"{'='*70}")
    print()
    print(df.to_string(index=False))
    print()


if __name__ == '__main__':
    main()
