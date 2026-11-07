"""CPU pilot: audit conversion semantics; not an MCU/TFLite deployment library."""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import io
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models import kan_model  # noqa: E402

BASE = "330395f7a9406e73d8501a68c9b2daae2747b6af"
MODEL_SPECS = [
    ("small_24_16", "24_16", "0.25"),
    ("full_24_16", "24_16", "1.0"),
    ("small_32_24_16", "32_24-16", "0.25"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_path(topology, width):
    name = (f"kan_{topology}_grid5_deg3_img224_bs256_lr0.003_wd1e-05_do0.05_"
            f"mobilenetv3_small_wm{width}")
    return ROOT / "experiment_data" / name / "models/kan_person_detector_best.pt"


def load_model(topology, width):
    path = checkpoint_path(topology, width)
    ck = torch.load(path, weights_only=True, map_location="cpu")
    config = ck["model_info"]["config"]
    k, p = config["kan"], config["preprocessor"]
    # Original constructor autosaves into ./model. Disable that side effect only.
    ctor = functools.partial(kan_model.KAN, auto_save=False, save_act=False)
    with patch.object(kan_model, "KAN", ctor):
        model = kan_model.KANImageClassifier(
            img_size=config["img_size"], input_channels=p["input_channels"],
            feature_dim=k["feature_dim"], kan_hidden_dims=k["hidden_dims"],
            kan_grid=k["grid"], kan_degree=k["degree"],
            use_batch_norm=k["use_batchnorm"], dropout_rate=k["dropout_rate"],
            activation_l1=k["activation_l1"], seed=k["seed"],
            preprocessor_type=p["preprocessor_type"], width_mult=p["width_mult"],
            preprocessor_pretrained=False,
        )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    metadata = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                "epoch": ck["epoch"], "historical_val_acc": ck["val_acc"],
                "config": config, "strict_load": True}
    return model, metadata


def historical_classes():
    """Compile the two *unchanged* class ASTs, avoiding TensorFlow imports."""
    path = ROOT / "scripts/convert_to_tflite.py"
    tree = ast.parse(path.read_text())
    names = {"FastLutKAN", "FastLutKANLayer"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if len(nodes) != 2:
        raise RuntimeError("Historical converter classes not found")
    env = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["FastLutKAN"], env["FastLutKANLayer"]


@torch.inference_mode()
def batched(model, x, batch_size=128):
    return torch.cat([model(b) for b in x.split(batch_size)])


@torch.inference_mode()
def exact_axis_mean(head, x):
    total = None
    for i in range(x.shape[1]):
        axis = torch.zeros_like(x)
        axis[:, i] = x[:, i]
        y = batched(head, axis)
        total = y if total is None else total + y
    return total / x.shape[1]


@torch.inference_mode()
def calibrate(head, x, margin=0.05):
    """Bounds use FP32 reference activations, with no evaluation data."""
    lows, highs = [], []
    for chunk in x.split(128):
        head(chunk)
        acts = head.kan.acts[:-1]
        if not lows:
            lows = [a.amin(0).clone() for a in acts]
            highs = [a.amax(0).clone() for a in acts]
        else:
            lows = [torch.minimum(lo, a.amin(0)) for lo, a in zip(lows, acts)]
            highs = [torch.maximum(hi, a.amax(0)) for hi, a in zip(highs, acts)]
    bounds = []
    for lo, hi in zip(lows, highs):
        pad = (hi - lo).clamp_min(1e-3) * margin
        bounds.append((lo - pad, hi + pad))
    return bounds


class EdgeLUTLayer(nn.Module):
    def __init__(self, table, low, high, subscale, subbias, nodescale, nodebias,
                 quantized=False):
        super().__init__()
        # Table layout: input, grid point, output. Quantization per edge.
        if quantized:
            scale = table.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
            stored = (table / scale).round().clamp(-127, 127).to(torch.int8)
        else:
            stored, scale = table, torch.empty(0)
        for name, val in [("table", stored), ("qscale", scale), ("low", low),
                          ("high", high), ("subscale", subscale), ("subbias", subbias),
                          ("nodescale", nodescale), ("nodebias", nodebias)]:
            self.register_buffer(name, val.detach().clone())
        self.last_saturated = 0
        self.last_values = 0

    def forward(self, x):
        self.last_saturated = int(((x < self.low) | (x > self.high)).sum())
        self.last_values = x.numel()
        pos = ((x.clamp(self.low, self.high) - self.low) /
               (self.high - self.low)) * (self.table.shape[1] - 1)
        left = pos.floor().long().clamp(0, self.table.shape[1] - 2)
        frac = (pos - left).unsqueeze(-1)
        inputs = torch.arange(x.shape[1], device=x.device)[None, :]
        a, b = self.table[inputs, left], self.table[inputs, left + 1]
        if self.table.dtype == torch.int8:
            scale = self.qscale[:, 0, :][None, :, :]
            a, b = a.float() * scale, b.float() * scale
        y = ((1 - frac) * a + frac * b).sum(1)
        return (y * self.subscale + self.subbias) * self.nodescale + self.nodebias


class EdgeLUT(nn.Module):
    def __init__(self, head, bounds, points, quantized=False):
        super().__init__()
        if points < 2:
            raise ValueError("At least two points required")
        kan = head.kan
        if any(w[1] != 0 for w in kan.width):
            raise ValueError("Multiplication nodes are unsupported in this pilot")
        if kan.symbolic_enabled and any(bool(s.mask.count_nonzero()) for s in kan.symbolic_fun):
            raise ValueError("Active symbolic edges are unsupported in this pilot")
        if not torch.equal(kan.input_id.long(), torch.arange(kan.width_in[0])):
            raise ValueError("Nonidentity input selection is unsupported in this pilot")
        if len(bounds) != len(kan.act_fun):
            raise ValueError("One range required per layer")
        layers = []
        with torch.inference_mode():
            for i, (layer, (lo, hi)) in enumerate(zip(kan.act_fun, bounds)):
                if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (hi > lo).all():
                    raise ValueError("Invalid calibration bounds")
                x = lo[None, :] + torch.linspace(0, 1, points)[:, None] * (hi - lo)[None, :]
                # postacts contains independent edge outputs, not whole-layer sections.
                _, _, postacts, _ = layer(x)
                table = postacts.permute(2, 0, 1).contiguous()
                layers.append(EdgeLUTLayer(table, lo, hi, kan.subnode_scale[i],
                    kan.subnode_bias[i], kan.node_scale[i], kan.node_bias[i], quantized))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def storage(self):
        table = sum(l.table.numel() * l.table.element_size() for l in self.layers)
        total = sum(b.numel() * b.element_size() for b in self.buffers())
        return {"table_bytes": table, "metadata_bytes": total - table,
                "stored_bytes": total}


@torch.inference_mode()
def evaluate(model, x):
    outputs, saturated, counts = [], None, None
    for b in x.split(128):
        outputs.append(model(b))
        if isinstance(model, EdgeLUT):
            if saturated is None:
                saturated = [0] * len(model.layers)
                counts = [0] * len(model.layers)
            for i, layer in enumerate(model.layers):
                saturated[i] += layer.last_saturated
                counts[i] += layer.last_values
    stats = {} if saturated is None else {"saturation_by_layer":
        [a / b for a, b in zip(saturated, counts)]}
    return torch.cat(outputs), stats


def metrics(y, reference):
    error = (y - reference).double()
    rmse = error.square().mean().sqrt().item()
    rms = reference.double().square().mean().sqrt().item()
    return {"rmse": rmse, "normalized_rmse": rmse / max(rms, 1e-12),
            "max_abs_error": error.abs().max().item(),
            "disagreement_count": int((y.argmax(1) != reference.argmax(1)).sum()),
            "disagreement_pct": float((y.argmax(1) != reference.argmax(1)).float().mean() * 100),
            "n": len(y)}


def quiet_construct(ctor, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        return ctor(*args, **kwargs).eval()
