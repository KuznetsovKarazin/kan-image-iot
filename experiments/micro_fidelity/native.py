"""Generate and verify a portable C99 FP32 evaluator with INT8 edge-table storage.

This is not fully integer execution or an ESP32 benchmark. Generated build
products are temporary; the generator plus calibrated model reproduces them.
"""
import ctypes
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


def c_array(name, data, integer=False):
    values = np.asarray(data).ravel()
    values = [str(int(x)) if integer else f"{float(x):.9e}f" for x in values]
    kind = "int8_t" if integer else "float"
    return f"static const {kind} {name}[] = {{" + ",".join(values) + "};\n"


def validate_native(lut, x, reference, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    layers = list(lut.layers)
    if any(l.table.dtype != torch.int8 for l in layers):
        raise ValueError("This native pilot expects INT8 stored tables")
    max_width = max(max(l.table.shape[0], l.table.shape[2]) for l in layers)
    text = "#include <stdint.h>\n#include <math.h>\n#include <stddef.h>\n"
    calls = []
    for n, l in enumerate(layers):
        table = l.table.cpu().numpy()
        for name in ["table", "qscale", "low", "high", "subscale", "subbias", "nodescale", "nodebias"]:
            value = getattr(l, name).cpu().numpy()
            text += c_array(f"l{n}_{name}", value, name == "table")
        ins, grid, outs = table.shape
        calls.append(f"layer(a,b,{ins},{outs},{grid},l{n}_table,l{n}_qscale,"
                     f"l{n}_low,l{n}_high,l{n}_subscale,l{n}_subbias,l{n}_nodescale,l{n}_nodebias);"
                     f" for(int j=0;j<{outs};++j) a[j]=b[j];\n")
    text += r'''
static void layer(const float *x, float *y, int ni, int no, int grid,
                  const int8_t *tab, const float *qs, const float *lo, const float *hi,
                  const float *ss, const float *sb, const float *ns, const float *nb) {
    for (int j=0;j<no;++j) y[j]=0.0f;
    for (int i=0;i<ni;++i) {
        float v=fminf(hi[i],fmaxf(lo[i],x[i]));
        float p=((v-lo[i])/(hi[i]-lo[i]))*(grid-1);
        int k=(int)floorf(p);
        if (k<0) k=0;
        if (k>grid-2) k=grid-2;
        float f=p-k;
        for (int j=0;j<no;++j) {
            float scale=qs[i*no+j];
            float a=(float)tab[(i*grid+k)*no+j]*scale;
            float b=(float)tab[(i*grid+k+1)*no+j]*scale;
            y[j]+=(1.0f-f)*a+f*b;
        }
    }
    for (int j=0;j<no;++j) y[j]=(y[j]*ss[j]+sb[j])*ns[j]+nb[j];
}
'''
    ni, no = layers[0].table.shape[0], layers[-1].table.shape[2]
    text += f"void infer(const float *x, float *y) {{ float a[{max_width}], b[{max_width}];\n"
    text += f"for(int i=0;i<{ni};++i) a[i]=x[i];\n" + "".join(calls)
    text += f"for(int j=0;j<{no};++j) y[j]=a[j];\n}}\n"
    text += f"void infer_batch(const float *x,float *y,int n){{for(int i=0;i<n;++i) infer(x+i*{ni},y+i*{no});}}\n"
    src, binary = workdir / "edge.c", workdir / "edge.so"
    src.write_text(text)
    flags = ["-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-shared", "-fPIC"]
    subprocess.run(["cc", *flags, str(src), "-lm", "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    lib = ctypes.CDLL(str(binary.resolve()))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.infer_batch.argtypes = [ptr, ptr, ctypes.c_int]
    lib.infer_batch.restype = None
    xx = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    yy = np.zeros((len(xx), no), np.float32)
    fn = lambda: lib.infer_batch(xx.ctypes.data_as(ptr), yy.ctypes.data_as(ptr), len(xx))
    for _ in range(5):
        fn()
    timings = []
    for _ in range(30):
        t = time.perf_counter_ns()
        fn()
        timings.append((time.perf_counter_ns() - t) / len(xx) / 1e3)
    delta = yy - reference.numpy()
    max_error = float(np.abs(delta).max())
    # Float summation ordering differs from PyTorch. Check logits and decisions.
    np.testing.assert_allclose(yy, reference.numpy(), atol=2e-5, rtol=2e-5)
    decisions = int(np.count_nonzero(yy.argmax(1) != reference.numpy().argmax(1)))
    if decisions:
        raise AssertionError(f"Native conversion changed {decisions} class decisions")
    info = {"max_abs_error_vs_python_lut": max_error,
            "rmse_vs_python_lut": float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
            "decision_disagreements_vs_python_lut": decisions,
            "n": len(xx), "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
            "compiler_flags": flags, "cpu_us_per_head_median": float(np.median(timings)),
            "cpu_us_per_head_p05": float(np.percentile(timings,5)),
            "cpu_us_per_head_p95": float(np.percentile(timings,95)),
            "timing_repetitions":30, "timing_scope":"hot-loop head-only CPU, no CNN, no MCU",
            "explicit_activation_buffers_bytes": 2*max_width*4,
            "buffers_note":"two declared arrays only, not measured total stack or device SRAM"}
    return torch.from_numpy(yy.copy()), info
