"""
TFLite Conversion Script for KAN-based Image Classifier.
Converts PyTorch model to TensorFlow Lite format for IoT deployment.

Author: Daniele Faggi
Date: February 2026

Conversion Pipeline: PyTorch → ONNX → TensorFlow → TFLite
"""
import time
from sympy.core.sympify import converter
import sys
from unittest.mock import MagicMock

# Mock matplotlib to avoid DLL errors on Windows (it's not needed for conversion)
# We must mock it BEFORE it is imported by kan
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.colors'] = MagicMock()
sys.modules['matplotlib.ticker'] = MagicMock()
sys.modules['matplotlib.transforms'] = MagicMock()
sys.modules['matplotlib._path'] = MagicMock()

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import random
from PIL import Image

# Add src and root to path for imports
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.kan_model import KANImageClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config  # Import project configuration

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import helper, numpy_helper

def surgical_fix_hardsigmoid(onnx_model_path, output_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Dizionario per tenere traccia dei nuovi nodi
    new_nodes = []
    
    for node in graph.node:
        if node.op_type == 'HardSigmoid':
            input_name = node.input[0]
            output_name = node.output[0]
            
            # Creiamo nomi univoci per i tensori intermedi
            add_out = output_name + "_add"
            mul_out = output_name + "_mul"
            relu_out = output_name + "_relu"
            
            # 1. Nodo ADD (x + 3.0)
            const_3 = helper.make_tensor(output_name + "_3", onnx.TensorProto.FLOAT, [], [3.0])
            graph.initializer.append(const_3)
            add_node = helper.make_node('Add', [input_name, output_name + "_3"], [add_out])
            
            # 2. Nodo MUL (x * 1/6)
            const_inv6 = helper.make_tensor(output_name + "_inv6", onnx.TensorProto.FLOAT, [], [0.16666666666666666])
            graph.initializer.append(const_inv6)
            mul_node = helper.make_node('Mul', [add_out, output_name + "_inv6"], [mul_out])
            
            # 3. Nodo ReLU (taglia a 0)
            relu_node = helper.make_node('Relu', [mul_out], [relu_out])
            
            # 4. Nodo MIN (taglia a 1)
            const_1 = helper.make_tensor(output_name + "_1", onnx.TensorProto.FLOAT, [], [1.0])
            graph.initializer.append(const_1)
            min_node = helper.make_node('Min', [relu_out, output_name + "_1"], [output_name])
            
            new_nodes.extend([add_node, mul_node, relu_node, min_node])
        else:
            new_nodes.append(node)
            
    # Sostituiamo i nodi nel grafo
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Pulizia e salvataggio
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Chirurgia completata! Modello salvato in: {output_path}")


class FastLutKAN(nn.Module):
    def __init__(self, original_kan, grid_range=[-2.2, 2.2], grid_size=256):
        super(FastLutKAN, self).__init__()
        self.grid_size = grid_size
        self.min_val = grid_range[0]
        self.max_val = grid_range[1]
        
        # Estraiamo la dimensione di input (il primo elemento di width)
        # width è una lista tipo [16, 32, 2]
        if isinstance(original_kan.kan.width[0], list):
            self.input_dim = original_kan.kan.width[0][0]
        else:
            self.input_dim = original_kan.kan.width[0]
            
        print(f"[INFO] Inizializzazione LUT per KAN con input_dim: {self.input_dim}")

        with torch.no_grad():
            x_base = torch.linspace(self.min_val, self.max_val, grid_size)
            # Creiamo una LUT che ha forma (grid_size, input_dim, output_dim)
            # Invece di una tabella piatta, abbiamo una tabella per ogni ingresso
            lut_list = []
            for i in range(self.input_dim):
                # Creiamo un input dove solo la i-esima feature varia
                x_test = torch.zeros(grid_size, self.input_dim)
                x_test[:, i] = x_base
                lut_list.append(original_kan(x_test).unsqueeze(1)) # (grid_size, 1, out)
            
            self.lut_table = nn.Parameter(torch.cat(lut_list, dim=1).detach())
            print(f"[OK] LUT multi-dim generata: {self.lut_table.shape}") 
            # Dovrebbe essere (256, 16, 4) se hai 4 classi
        """
        with torch.no_grad():
            # Generiamo la griglia di campionamento
            x_base = torch.linspace(self.min_val, self.max_val, grid_size)
            
            # Creiamo l'input di test (grid_size, input_dim)
            # Immaginiamo di "scansionare" la risposta della rete per ogni possibile valore di input
            x_test = x_base.unsqueeze(1).repeat(1, self.input_dim)
            
            # Calcoliamo la risposta dell'intera KAN (tutti i layer)
            # Questo "congela" la logica MultKAN in una tabella finale
            self.lut_table = nn.Parameter(original_kan(x_test).detach())
            print(f"[OK] LUT generata. Shape tabella: {self.lut_table.shape}")
        """
    def forward(self, x):
        # x shape: (Batch, 16)
        grid_step = (self.max_val - self.min_val) / (self.grid_size - 1)
        
        # 1. Portiamo x in "unità di griglia"
        x_grid = ((x - self.min_val) / grid_step).unsqueeze(-1) # (Batch, 16, 1)

        # 2. Creiamo la griglia di indici
        grid = torch.arange(self.grid_size, device=x.device).float().view(1, 1, -1) # (1, 1, 256)

        # 3. Calcoliamo la maschera di pesi (Batch, 16, 256)
        dist = torch.abs(x_grid - grid)
        weight_mask = torch.relu(1.0 - dist)

        # 4. ALLINEAMENTO PER IL PRODOTTO (Il punto critico)
        # Dobbiamo moltiplicare ogni feature (1..16) per la sua LUT specifica.
        # weight_mask: (Batch, 16, 256) -> la ruotiamo per mettere la griglia in mezzo
        # mask_permuted: (Batch, 256, 16)
        mask_permuted = weight_mask.permute(0, 2, 1)

        # self.lut_table è (256, 16, 2)
        # Moltiplichiamo elemento per elemento: (Batch, 256, 16) * (1, 256, 16)
        # Poi aggiungiamo una dimensione vuota alla maschera per includere le classi (Out=2)
        # mask_expanded: (Batch, 256, 16, 1)
        # lut_expanded:  (1, 256, 16, 2)
        combined = mask_permuted.unsqueeze(-1) * self.lut_table.unsqueeze(0)
        
        # 5. Riduzione finale
        # Sommiamo sulla dimensione della griglia (dim 1) per ottenere (Batch, 16, 2)
        res = combined.sum(dim=1)
        
        # Media delle feature
        return res.mean(dim=1)
"""
    def forward(self, x):
        # x shape: (Batch, input_dim)
        # 1. Portiamo l'input nel range [0, grid_size - 1]
        x_norm = (x - self.min_val) / (self.max_val - self.min_val)
        x_scaled = x_norm * (self.grid_size - 1)
        
        # 2. Indici per il lookup (clamp per sicurezza)
        indices = x_scaled.int().clamp(0, self.grid_size - 1)
        
        # 3. Lookup. Se x è (B, 16), prendiamo il valore medio della LUT
        # Nota: Essendo una LUT 1D approssimata, usiamo la media delle feature 
        # o la mappatura diretta se la rete è lineare.
        # Per semplicità in ONNX usiamo il primo indice o la media:
        idx = indices[:, 0] # Prendiamo la prima feature come riferimento per la LUT
        
        return torch.index_select(self.lut_table, 0, idx)
"""

def export_to_onnx(model, output_path, img_size=224, use_lut=False):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Path to save ONNX model
        img_size: Input image size (default: 224)
    """
    print("\n" + "="*60)
    print("STEP 1: Exporting PyTorch model to ONNX")
    print("="*60)
    

    # Set model to eval mode
    model.eval()

    # Tentativo di conversione KAN in LUT
    if use_lut:
        model.kan = FastLutKAN(model.kan, grid_range=[-2.0, 2.0], grid_size=16)

    # Subst relu0to1 with ReLU
    #replace_relu0to1(model.preprocessor)
    #replace_hardsigmoid(model.preprocessor)
    #time.sleep(5)
    # Applica al modello globale che contiene preprocessor, projector e kan
    #deep_clean_everything(model)
    #time.sleep(5)

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path + ".orig.onnx",
            export_params=True,
            opset_version=13,  # Use stable opset version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"[OK] Successfully exported to ONNX: {output_path}")
        
        # Verify ONNX model
        import onnx
        surgical_fix_hardsigmoid(output_path + ".orig.onnx", output_path)
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")
        
        return True
    except Exception as e:
        print(f"[FAIL] Failed to export to ONNX: {e}")
        return False


def load_calibration_dataset(img_size=224, samples=20):
    """
    Load real calibration data from the project's data directory.
    Target path: data/processed/vww_subset/val
    """
    data_root = os.path.join(os.getcwd(), 'data', 'processed', 'vww_subset', 'val')
    if not os.path.exists(data_root):
        print(f"[WARN] Calibration data not found at {data_root}")
        return None

    print(f"[INFO] Loading calibration images from {data_root}...")
    
    # Collect image paths from both classes
    image_paths = []
    for class_name in ['person', 'no_person']:
        class_dir = os.path.join(data_root, class_name)
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_paths.extend(files)
            
    if not image_paths:
        print("[WARN] No images found in calibration directory")
        return None
        
    print(f"[INFO] Found {len(image_paths)} images. Using {min(samples, len(image_paths))} for calibration.")
    random.shuffle(image_paths)
    image_paths = image_paths[:samples]
    
    # Preprocessing transforms (Standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            # Standard TFLite/ONNX2TF expects NHWC: (1, 224, 224, 3)
            # PyTorch's Transform gives NCHW: (3, 224, 224)
            # Let's transpose to NHWC: (224, 224, 3)
            tensor = tensor.permute(1, 2, 0)
            # Add batch dimension: (1, 224, 224, 3)
            tensor = tensor.unsqueeze(0)
            images.append(tensor)
        except Exception as e:
            print(f"Error loading {os.path.basename(img_path)}: {e}")
            
    if not images:
        return None
        
    # Stack into a generator
    def representative_dataset():
        for img_tensor in images:
            # Yield as list of inputs (TFLite converter expects list)
            yield [img_tensor.numpy().astype(np.float32)]
            
    return representative_dataset


def convert_pytorch_to_tflite_direct(model, output_path, img_size=224, quantize='none'):
    """
    Convert PyTorch model directly to TFLite using ai_edge_torch.
    This is the modern Google-supported approach.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Path to save TFLite model
        img_size: Input image size
        quantize: Quantization type ('none', 'float16', 'int8')
    """
    print("\n" + "="*60)
    print("STEP 2: Converting PyTorch to TFLite (direct conversion)")
    print("="*60)
    
    try:
        import ai_edge_torch
        
        # Create sample input
        sample_input = (torch.randn(1, 3, img_size, img_size),)
        
        import tensorflow as tf
        
        print("Converting with ai_edge_torch...")
        
        # Determine quantization configuration
        quant_config = None
        if quantize == 'float16':
            print("Applying float16 quantization...")
            # ai_edge_torch handles this during conversion
        elif quantize == 'int8':
            print("Applying int8 quantization...")
            # Try to load real calibration data
            calibration_loader = load_calibration_dataset(img_size)
            if calibration_loader:
                quant_config = ai_edge_torch.quantize.QuantConfig(
                    calibration_loader=calibration_loader
                )
            else:
                print("[WARN] Using random data for calibration (suboptimal for int8)")
                # Fallback to random data
                def representative_dataset():
                    for _ in range(100):
                        data = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
                        # Normalize roughly like real data
                        data = (data - 0.45) / 0.225 
                        yield [data]
                quant_config = ai_edge_torch.quantize.QuantConfig(
                    calibration_loader=representative_dataset
                )
        
        edge_model = ai_edge_torch.convert(
            model, 
            (dummy_input,),
            quant_config=quant_config
        )
        
        edge_model.export(output_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] Successfully converted to TFLite: {output_path}")
        print(f"  Model size: {file_size:.2f} MB")
        
        return True, file_size
        
    except ImportError:
        print("[SKIP] ai_edge_torch not installed or failed to import.")
        print("  To use direct conversion: pip install ai-edge-torch")
        print("  Note: This library currently requires Linux (or WSL).")
        return False, 0.0
    except Exception as e:
        print(f"[FAIL] Direct conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def convert_onnx_to_tensorflow(onnx_path, tf_output_path):
    """
    Convert ONNX model to TensorFlow SavedModel format using onnx2tf.
    
    Args:
        onnx_path: Path to ONNX model
        tf_output_path: Path to save TensorFlow SavedModel
    """
    print("\n" + "="*60)
    print("STEP 2: Converting ONNX to TensorFlow SavedModel")
    print("="*60)
    
    try:
        # Try using onnx2tf (modern alternative to onnx-tf)
        import subprocess
        import sys
        
        # Use onnx2tf command line tool via python module
        print(f"Running onnx2tf via: {sys.executable} -m onnx2tf")
        
        cmd = [sys.executable, '-m', 'onnx2tf', '-i', onnx_path, '-o', tf_output_path]
        
        # Run conversion
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        print("onnx2tf output:")
        for line in process.stdout:
            print("  " + line.strip())
            
        return_code = process.wait()
        
        if return_code == 0:
            print(f"[OK] Successfully converted to TensorFlow: {tf_output_path}")
            return True
        else:
            print(f"[WARN] onnx2tf failed with return code {return_code}")
            
            # Check if auto-correction JSON was generated
            # onnx2tf generates model_auto.json INSIDE the output directory
            json_path = os.path.join(tf_output_path, "model_auto.json")
            if os.path.exists(json_path):
                print(f"[INFO] Auto-correction JSON found: {json_path}")
                print("Retrying conversion with auto-correction...")
                
                cmd_retry = cmd + ['-prf', json_path]
                
                process_retry = subprocess.Popen(
                    cmd_retry,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                print("onnx2tf retry output:")
                for line in process_retry.stdout:
                    print("  " + line.strip())
                    
                return_code_retry = process_retry.wait()
                
                if return_code_retry == 0:
                    print(f"[OK] Successfully converted to TensorFlow (with auto-correction): {tf_output_path}")
                    return True
                else:
                    print(f"[FAIL] onnx2tf retry failed with return code {return_code_retry}")
                    raise Exception("onnx2tf conversion failed after retry")
            else:
                raise Exception("onnx2tf conversion failed")
            
    except ImportError:
        print("[INFO] subprocess module not found (unlikely)")
    except Exception as e:
        print(f"[INFO] onnx2tf failed: {e}")
    
    # Fallback: try onnx-tf if available
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export as SavedModel
        tf_rep.export_graph(tf_output_path)
        
        print(f"[OK] Successfully converted to TensorFlow: {tf_output_path}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to convert ONNX to TensorFlow: {e}")
        print("\nPlease install one of the following:")
        print("  pip install onnx2tf  (recommended)")
        print("  pip install onnx-tf tensorflow-probability  (legacy)")
        return False


def convert_tensorflow_to_tflite(tf_model_path, tflite_output_path, 
                                  quantize='none', representative_dataset=None,
                                  enforce_quantization = True):
    """
    Convert TensorFlow SavedModel to TFLite format.
    
    Args:
        tf_model_path: Path to TensorFlow SavedModel
        tflite_output_path: Path to save TFLite model
        quantize: Quantization type ('none', 'float16', 'int8')
        representative_dataset: Dataset for calibration (required for int8)
    """
    print("\n" + "="*60)
    print(f"STEP 3: Converting TensorFlow to TFLite (quantize={quantize})")
    print("="*60)
    
    try:
        import tensorflow as tf
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Apply quantization
        if quantize == 'float16':
            if enforce_quantization:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_FLOAT16]
                converter.inference_input_type = tf.float16
                converter.inference_output_type = tf.float16

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("Applying float16 quantization...")

        elif quantize == 'int8':
            if enforce_quantization:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                print("Applying int8 quantization with calibration...")
            else:
                print("[WARN] Warning: int8 quantization without calibration dataset")
        else:
            print("No quantization applied (float32)")
        
        # Convert
        tflite_model = converter.convert()
        
        # Save to file
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file size
        file_size_mb = os.path.getsize(tflite_output_path) / (1024 * 1024)
        print(f"[OK] Successfully converted to TFLite: {tflite_output_path}")
        print(f"  Model size: {file_size_mb:.2f} MB")
        
        return True, file_size_mb
    except Exception as e:
        print(f"[FAIL] Failed to convert to TFLite: {e}")
        print("\nNote: Make sure you have installed TensorFlow:")
        print("  pip install tensorflow")
        return False, 0


def representative_dataset_gen(dataloader, num_samples=100):
    """
    Generator for representative dataset (for int8 quantization calibration).
    
    Args:
        dataloader: PyTorch DataLoader
        num_samples: Number of samples to use for calibration
    """
    count = 0
    for images, _ in dataloader:
        # Convert PyTorch tensor to numpy
        batch = images.numpy()
        for img in batch:
            if count >= num_samples:
                return
            # Expand dims to match input shape [1, H, W, C]
            yield [np.expand_dims(img.transpose(1, 2, 0), axis=0).astype(np.float32)]
            count += 1


def verify_tflite_model(tflite_path, pytorch_model, dataloader, img_size=224):
    """
    Verify TFLite model by comparing outputs with PyTorch model.
    
    Args:
        tflite_path: Path to TFLite model
        pytorch_model: Original PyTorch model
        dataloader: Test dataloader
        img_size: Input image size
    """
    print("\n" + "="*60)
    print("STEP 4: Verifying TFLite model")
    print("="*60)
    
    try:
        import tensorflow as tf
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Test on a batch
        pytorch_model.eval()
        images, labels = next(iter(dataloader))
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = pytorch_model(images)
            pytorch_preds = torch.argmax(pytorch_outputs, dim=1).numpy()
        
        # TFLite inference
        tflite_preds = []
        for i in range(min(10, len(images))):  # Test first 10 images
            # Prepare input (convert from [C, H, W] to [H, W, C])
            img_np = images[i].numpy().transpose(1, 2, 0)
            img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_np)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Get prediction
            pred = np.argmax(output[0])
            tflite_preds.append(pred)
        
        tflite_preds = np.array(tflite_preds)
        
        # Compare predictions
        agreement = np.mean(pytorch_preds[:10] == tflite_preds) * 100
        print(f"\nPrediction agreement (first 10 samples): {agreement:.1f}%")
        
        if agreement >= 90:
            print("[OK] TFLite model verification PASSED")
            return True
        else:
            print("[WARN] Warning: Low agreement between PyTorch and TFLite models")
            print("  This might be due to quantization or conversion artifacts")
            return False
            
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert KAN Image Classifier from PyTorch to TFLite'
    )
    parser.add_argument('model_path', type=str, nargs='?', default=None,
                        help='Path to PyTorch checkpoint (.pt file). Defaults to current experiment best model.')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output TFLite filename (default: model.tflite in experiment dir)')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'float16', 'int8'],
                        help='Quantization type (default: none)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify converted model (default: True)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--no_verify', dest='verify', action='store_false',
                        help='Skip verification step')
    parser.add_argument('--force_config', dest='force_config', action='store_true',
                        help='Force configuration from config.py')    
    parser.add_argument('--use_lut', dest='use_lut', action='store_true',
                        help='Use LUT for KAN (default: False)')  
    parser.add_argument('--enforce_quantization', dest='enforce_quantization', action='store_true',
                        help='Enforce quantization (default: False)')  
    args = parser.parse_args()
    
    # Handle default model path
    if args.model_path is None:
        paths = config.get_experiment_paths()
        default_model = paths['model_dir'] / 'kan_person_detector_best.pt'
        args.model_path = str(default_model)
        print(f"\n[INFO] No model path provided. Using default from config: {args.model_path}")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    print("="*60)
    print("KAN Image Classifier - PyTorch to TFLite Conversion")
    print("="*60)
    print(f"Input model: {args.model_path}")
    print(f"Output file: {args.output_name}")
    print(f"Quantization: {args.quantize}")
    print(f"Verification: {args.verify}")
    
    # Create output directory
    if args.output_name is None:
        paths = config.get_experiment_paths()
        output_dir = paths['model_dir']
        tflite_filename = 'model.tflite'
        print(f"Output directory (auto): {output_dir}")
    else:
        # User specified output path
        output_path_user = Path(args.output_name)
        output_dir = output_path_user.parent
        if str(output_dir) == '.':
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        tflite_filename = output_path_user.name
        print(f"Output directory (custom): {output_dir}")
    
    # Define intermediate paths
    onnx_path = output_dir / "model.onnx"
    tf_path = output_dir / "tf_model"
    tflite_path = output_dir / tflite_filename
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    #  Try to get configuration from checkpoint first, fallback to config.py
    #  Try to get configuration from checkpoint first, fallback to config.py
    saved_config = None
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
        elif 'model_info' in checkpoint and 'config' in checkpoint['model_info']:
            saved_config = checkpoint['model_info']['config']

    if saved_config and not args.force_config:
        print("\nUsing configuration from checkpoint...")
        
        if 'preprocessor' in saved_config:
            # Checkpoint contains preprocessor config
            preproc_config = saved_config['preprocessor']
            kan_config = saved_config['kan']
            
            feature_dim = preproc_config['output_features']
            hidden_dims = kan_config.get('hidden_dims', [feature_dim // 2])
            grid = kan_config.get('grid', 5)
            degree = kan_config.get('degree', 3)
            
            # Infer preprocessor type
            if 'preprocessor_type' in preproc_config:
                preprocessor_type = preproc_config['preprocessor_type']
            else:
                # If missing, it's likely the MobileNet version (common in this project)
                # simpler CNN was earlier prototype
                print("[INFO] preprocessor_type missing in config, assuming 'mobilenetv3_small'")
                preprocessor_type = 'mobilenetv3_small'
                
            width_mult = preproc_config.get('width_mult', 1.0)
        else:
            # Try modern format (MobileNet preprocessor with KAN config)
            print("[WARN] Warning: config format not recognized, falling back to config.py")
            feature_dim = config.PREPROCESSOR_CONFIG['output_features']
            hidden_dims = config.KAN_CONFIG['hidden_dims']
            grid = config.KAN_CONFIG['grid']
            degree = config.KAN_CONFIG['degree']
            preprocessor_type = config.PREPROCESSOR_CONFIG['preprocessor_type']
            width_mult = config.PREPROCESSOR_CONFIG['width_mult']
    else:
        print("\nUsing configuration from config.py...")
        feature_dim = config.PREPROCESSOR_CONFIG['output_features']
        hidden_dims = config.KAN_CONFIG['hidden_dims']
        grid = config.KAN_CONFIG['grid']
        degree = config.KAN_CONFIG['degree']
        preprocessor_type = config.PREPROCESSOR_CONFIG['preprocessor_type']
        width_mult = config.PREPROCESSOR_CONFIG['width_mult']
    
    print(f"  Preprocessor type: {preprocessor_type}")
    print(f"  Width multiplier: {width_mult}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  KAN hidden dims: {hidden_dims}")
    print(f"  KAN grid: {grid}")
    print(f"  KAN degree: {degree}")
    
    # Create model with detected configuration
    model = KANImageClassifier(
        input_channels=config.PREPROCESSOR_CONFIG.get('input_channels', 3),
        img_size=args.img_size,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        preprocessor_type=preprocessor_type,
        width_mult=width_mult,
        preprocessor_pretrained=config.PREPROCESSOR_CONFIG.get('pretrained', False),
        preprocessor_freeze=False,  # Not relevant for inference
    )
    
    # Load weights - handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Standard training checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[OK] Loaded weights from training checkpoint")
        elif 'quantized_state' in checkpoint:
            # Quantized model checkpoint
            model.load_state_dict(checkpoint['quantized_state'])
            print("[OK] Loaded weights from quantized checkpoint")
            print(f"  Original FP32 size: {checkpoint.get('fp32_size_mb', 'N/A')} MB")
            print(f"  Quantized size: {checkpoint.get('hybrid_size_mb', 'N/A')} MB")
        elif 'state_dict' in checkpoint:
            # Alternative checkpoint format
            model.load_state_dict(checkpoint['state_dict'])
            print("[OK] Loaded weights from checkpoint")
        else:
            # Try loading the checkpoint directly as state_dict
            try:
                model.load_state_dict(checkpoint)
                print("[OK] Loaded weights directly from state_dict")
            except Exception as e:
                print(f"[FAIL] Could not load checkpoint. Available keys: {list(checkpoint.keys())}")
                raise e
    else:
        # Checkpoint is directly a state_dict
        model.load_state_dict(checkpoint)
        print("[OK] Loaded weights directly from state_dict")
    
    model.eval()
    
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Try direct PyTorch to TFLite conversion first (faster and more reliable)
    print("\n" + "="*60)
    print("Attempting direct PyTorch -> TFLite conversion...")
    print("="*60)
    
    success, file_size = convert_pytorch_to_tflite_direct(
        model, str(tflite_path), args.img_size, args.quantize
    )
    
    # If direct conversion failed, fall back to ONNX→TF→TFLite path
    if not success:
        print("\n" + "="*60)
        print("Falling back to ONNX -> TensorFlow -> TFLite path...")
        print("="*60)
        
        # Step 1: Export to ONNX
        if not export_to_onnx(model, str(onnx_path), args.img_size, args.use_lut):
            print("\n[FAIL] Conversion failed at ONNX export stage")
            sys.exit(1)
        
        # Step 2: Convert ONNX to TensorFlow
        if not convert_onnx_to_tensorflow(str(onnx_path), str(tf_path)):
            print("\n[FAIL] Conversion failed at TensorFlow conversion stage")
            sys.exit(1)
        
        # Prepare representative dataset for int8 quantization
        # Prepare representative dataset for int8 quantization
        rep_dataset = None
        if args.quantize == 'int8':
            print("\nPreparing calibration dataset for int8 quantization...")
            rep_dataset = load_calibration_dataset(args.img_size)
            
            if rep_dataset:
                print("[OK] Calibration dataset prepared")
            else:
                print("[WARN] Calibration dataset not found, falling back to random data (suboptimal)")
                def random_dataset_gen():
                    for _ in range(100):
                        data = np.random.rand(1, 3, args.img_size, args.img_size).astype(np.float32)
                        data = (data - 0.45) / 0.225
                        yield [data.transpose(0, 2, 3, 1)] # TF expects NHWC
                rep_dataset = random_dataset_gen
        
        # Step 3: Convert TensorFlow to TFLite
        success, file_size = convert_tensorflow_to_tflite(
            str(tf_path), 
            str(tflite_path),
            args.quantize,
            rep_dataset,
            args.enforce_quantization
        )
        
        if not success:
            print("\n[FAIL] Conversion failed at TFLite conversion stage")
            sys.exit(1)
    
    # Step 4: Verify model (optional)
    if args.verify:
        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_path = Path('data/processed/vww_subset/test')
        if test_path.exists():
            test_dataset = datasets.ImageFolder(str(test_path), transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            verify_tflite_model(str(tflite_path), model, test_loader, args.img_size)
        else:
            print("[WARN] Warning: Test dataset not found, skipping verification")
    
    # Final summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"[OK] TFLite model saved to: {tflite_path}")
    print(f"  Model size: {file_size:.2f} MB")
    print(f"  Quantization: {args.quantize}")
    
    if file_size <= 1.5:
        print(f"\n[OK] Model size is IoT-ready (<= 1.5 MB)")
    elif file_size <= 3.0:
        print(f"\n[WARN] Model size is acceptable for some IoT devices (<= 3 MB)")
    else:
        print(f"\n[WARN] Model may be too large for resource-constrained IoT devices")
        print(f"  Consider using quantization (--quantize float16 or --quantize int8)")
    
    print("\nNext steps:")
    print("  1. Test the TFLite model on your target IoT device")
    print("  2. Measure inference latency and memory usage")
    print("  3. Fine-tune quantization if needed for better performance")
    print("="*60)


if __name__ == '__main__':
    main()
