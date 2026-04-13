"""
KAN-based Image Classification Model for IoT Devices.
This module implements a lightweight Kolmogorov-Arnold Network architecture 
optimized for person detection on resource-constrained devices.

Author: Oleksandr Kuznetsov
Date: March 2025

Adaptations for using MobileNetV2 as feature extractor.

Changed by: Daniele Faggi
Date: September 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_patched import KAN
import numpy as np
import torchvision

class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth regularization technique.
    During training, randomly drops entire layers with probability p.
    """
    def __init__(self, drop_prob=0.0):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * random_tensor / keep_prob

class KANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, grid=5, k=3):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Calcoliamo la dimensione dell'input per la KAN interna
        # Ogni patch 3x3 ha (in_channels * 3 * 3) valori
        kan_input_dim = in_channels * kernel_size * kernel_size
        
        # Inizializziamo la TUA classe KAN
        # Questa è la parte che poi convertirai in LUT post-training
        self.kan = KAN(
            width=[kan_input_dim, out_channels], 
            grid=grid, 
            k=k
        )

    def forward(self, x):
        # 1. Dimensioni originali
        b, c, h, w = x.shape
        
        # 2. Unfold: Trasforma l'immagine in una sequenza di patch
        # Output: [Batch, in_channels * k * k, L] dove L è il numero di posizioni del kernel
        x_unfold = nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride
        )
        
        # 3. Prepariamo i dati per la KAN
        # Trasponiamo per avere [Batch * L, Features]
        # Questo permette alla KAN di processare ogni "posizione" del filtro in parallelo
        x_unfold = x_unfold.transpose(1, 2).contiguous()
        x_flat = x_unfold.view(-1, x_unfold.shape[-1])
        
        # 4. PASSAGGIO NELLA KAN (Qui avviene la magia non lineare)
        # Se hai attivato la modalità LUT, qui la KAN userà le tue tabelle
        out_flat = self.kan(x_flat)
        
        # 5. Ricomposizione dell'immagine (Fold)
        # Calcoliamo le nuove dimensioni H e W
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Riportiamo il tensore alla forma [Batch, out_channels, H, W]
        out = out_flat.view(b, out_h, out_w, self.out_channels)
        return out.permute(0, 3, 1, 2).contiguous()

    # Metodo di utilità per aggiornare la griglia (fondamentale per le KAN)
    def update_grid(self, x):
        x_unfold = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_unfold = x_unfold.transpose(1, 2).contiguous()
        x_flat = x_unfold.view(-1, x_unfold.shape[-1])
        self.kan.update_grid(x_flat)

class KKANBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Convoluzione KAN: è qui che applicherai la tua LUT post-training
        self.kan_conv = KANConv2d(in_ch, out_ch, kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        
        # Shortcut per la connessione residuale (fondamentale per VWW)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        # x -> KANConv -> BN -> + Shortcut
        # Nota: Non serve ReLU, la KAN è già non-lineare!
        out = self.bn(self.kan_conv(x))
        out += self.shortcut(x)
        return out

class KANDepthwiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, grid=5, k=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # La KAN opera su una patch "single channel" (es. 3x3 = 9 input)
        # Ne creiamo una che produce 1 output per ogni canale.
        # Per efficienza, processeremo i canali come un unico grande batch.
        self.kan = KAN(
            width=[kernel_size * kernel_size, 1], 
            grid=grid, 
            k=k
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. Unfold per canale: usiamo groups=c per isolare i canali
        # Invece di un unfold gigante, sfruttiamo il fatto che i canali sono indipendenti
        # Creiamo patch di dimensione [B, C, K*K, L]
        x_unfold = nn.functional.unfold(
            x, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride
        )
        
        # 2. Reshape strategico per risparmiare memoria
        # Portiamo i canali nella dimensione del batch per la KAN: 
        # [B * C * L, K*K]
        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        L = out_h * out_w
        
        x_unfold = x_unfold.view(b, c, self.kernel_size**2, L)
        x_unfold = x_unfold.permute(0, 1, 3, 2).contiguous() # [B, C, L, K*K]
        x_kan_input = x_unfold.view(-1, self.kernel_size**2) # [B*C*L, 9]
        
        # 3. Passaggio nella KAN (Qui la LUT lavora su vettori da 9)
        out_kan = self.kan(x_kan_input) # [B*C*L, 1]
        
        # 4. Ricomposizione
        out = out_kan.view(b, c, out_h, out_w)
        return out

class KKAN_MobileBlock(nn.Module):
    """
    Blocco stile MobileNet: Depthwise KAN + Pointwise Standard
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Estrazione spaziale non lineare (Depthwise KAN)
        self.depthwise = KANDepthwiseConv2d(in_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_ch)
        
        # Mix dei canali (Pointwise Lineare - Standard Conv 1x1)
        # Questo mantiene bassi i parametri e la memoria
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        
        return out + identity

class KAN_Activation_Conv(nn.Module):
    """
    Applica la logica KAN (Spline) come se fosse un'attivazione 
    su una convoluzione standard. Zero Unfold = Zero spreco di memoria.
    """
    def __init__(self, channels, grid=5, k=3):
        super().__init__()
        # Usiamo una KAN che accetta 1 input e dà 1 output (element-wise)
        # La applichiamo a ogni pixel in modo indipendente
        self.kan_act = RegularizedKAN([1, 1], grid=grid, degree=k, dropout_rate=0.1, activation_l1=1e-5)

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape
        # Flatten spaziale per la KAN: [B*C*H*W, 1]
        x_flat = x.view(-1, 1)
        # La KAN lavora pixel per pixel (come una ReLU evoluta)
        out_flat = self.kan_act(x_flat)
        return out_flat.view(b, c, h, w)

class KKAN_EfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 1. Depthwise Conv Standard (Estrae lo spazio linearmente)
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, 
                                 padding=1, stride=stride, groups=in_ch, bias=False)
        
        # 2. KAN Activation (La parte non lineare con LUT)
        # Qui la tua LUT lavora su singoli valori, non su patch da 9!
        self.kan_act = KAN_Activation_Conv(in_ch)
        
        # 3. Pointwise Standard
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.kan_act(x) # Qui avviene la magia KAN/LUT
        x = self.pw_conv(x)
        return self.bn(x)

"""
class KKAN_VWW_96(nn.Module):
    def __init__(self, output_features=128):
        super().__init__()
        
        # 1. Input Stem (Leggera): Porta i 3 canali RGB a 16 feature maps
        # Usiamo una Conv standard qui per non appesantire l'input 96x96
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16)
        )

        # 2. KKAN Layers (Il cuore del riconoscimento feature)
        # Stage 1: 96x96 -> 48x48
        #self.layer1 = KKAN_MobileBlock(16, 32, stride=2) 
        
        # Stage 2: 48x48 -> 24x24
        #self.layer2 = KKAN_MobileBlock(32, 64, stride=2)
        
        # Stage 3: 24x24 -> 12x12
        self.layer3 = KKAN_MobileBlock(16, output_features, stride=2)

        # 3. Global Header
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier finale: Una KAN MLP pura
        # Anche questa andrà in LUT
        #self.classifier = KAN([128, 64, num_classes], grid=5, k=3)

    def forward(self, x):
        x = self.stem(x)     # [B, 16, 96, 96]
        #x = self.layer1(x)   # [B, 32, 48, 48]
        #x = self.layer2(x)   # [B, 64, 24, 24]
        x = self.layer3(x)   # [B, 128, 12x12]
        
        x = self.gap(x).view(x.size(0), -1) # [B, 128]
        return x #self.classifier(x)
"""

class KKAN_VWW_96(nn.Module):
    def __init__(self, output_features=128):
        super().__init__()
        
        # 1. Input Stem (Leggera): Porta i 3 canali RGB a 16 feature maps
        # Usiamo una Conv standard qui per non appesantire l'input 96x96
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16)
        )

        # 2. KKAN Layers (Il cuore del riconoscimento feature)
        # Stage 1: 96x96 -> 48x48
        self.layer1 = KKAN_EfficientBlock(16, 32, stride=1) 
        
        # Stage 2: 48x48 -> 24x24
        self.layer2 = KKAN_EfficientBlock(32, 64, stride=2)
        
        # Stage 3: 24x24 -> 12x12
        self.layer3 = KKAN_EfficientBlock(64, output_features, stride=2)

        # 3. Global Header
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier finale: Una KAN MLP pura
        # Anche questa andrà in LUT
        #self.classifier = KAN([128, 64, num_classes], grid=5, k=3)

    def forward(self, x):
        x = self.stem(x)     # [B, 16, 96, 96]
        x = self.layer1(x)   # [B, 32, 48, 48]
        x = self.layer2(x)   # [B, 64, 24, 24]
        x = self.layer3(x)   # [B, 128, 12x12]
        
        x = self.gap(x).view(x.size(0), -1) # [B, 128]
        return x #self.classifier(x)


class MobileNetV2Preprocessor(nn.Module):
    """
    Preprocessing module using MobileNetV2 to convert image features to format suitable for KAN.
    Leverages pretrained MobileNetV2 for efficient feature extraction.
    """
    def __init__(self, output_features=48,  
                 pretrained=True, freeze_mobilenet=True):
        super(MobileNetV2Preprocessor, self).__init__()
        
        # Load pretrained MobileNetV2 model
        # self.mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)        
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.mobilenet.classifier = nn.Sequential()  # Rimuove il classificatore
        
        # Freeze MobileNetV2 parameters if specified
        if freeze_mobilenet:
            for param in self.mobilenet.parameters():
                param.requires_grad = False
        
        # Final feature dimension after MobileNetV2
        mobilenet_output_dim = 1280  # MobileNetV2 final feature dimension
        
        # Linear projection to desired output feature size
        self.projector = nn.Sequential(
            nn.Linear(mobilenet_output_dim, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # MobileNetV2 expects 3 channels, but input might be 1 or 3.
        # If input_channels is 1, we need to convert to 3 channels.
        if x.size(1) == 1:
            # Convert grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        features = self.mobilenet(x)
        # features should be [batch, 1280]
        return self.projector(features)

class MobileNetV3LitePreprocessor(nn.Module):
    """
    Preprocessing module using MobileNetV3 Lite to convert image features to format suitable for KAN.
    Leverages pretrained MobileNetV3 Lite for efficient feature extraction.
    """
    def __init__(self, output_features=48,
                 pretrained=True, freeze_mobilenet=True, width_mult=1.0, img_size=224):
        super(MobileNetV3LitePreprocessor, self).__init__()
        
        # Load pretrained MobileNetV3 model
        # self.mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)        
        self.mobilenet = torchvision.models.mobilenet_v3_small(pretrained=pretrained, width_mult=width_mult)
        self.mobilenet.classifier = nn.Sequential()  # Rimuove il classificatore

        # For small input images (≤128px) MobileNetV3-Small has 5×stride-2 layers,
        # which collapses the spatial resolution to 3×3 before GAP (only 9 positions).
        # Removing the first stride-2 keeps the final feature map at 6×6 (36 positions),
        # giving 4× more spatial context without touching the pretrained weights.
        if img_size <= 128:
            self.mobilenet.features[0][0].stride = (1, 1)

        # Freeze MobileNetV3 parameters if specified
        if freeze_mobilenet:
            for param in self.mobilenet.parameters():
                param.requires_grad = False
        
        # Final feature dimension after MobileNetV3
        mobilenet_output_dim = 576  # MobileNetV3 Lite final feature dimension
        mobilenet_output_dim = int(mobilenet_output_dim * width_mult) # Adjust for width multiplier

        # Linear projection to desired output feature size
        self.projector = nn.Sequential(
            nn.Linear(mobilenet_output_dim, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # MobileNetV2 expects 3 channels, but input might be 1 or 3.
        # If input_channels is 1, we need to convert to 3 channels.
        if x.size(1) == 1:
            # Convert grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        features = self.mobilenet(x)
        # features should be [batch, 1280]
        return self.projector(features)

class MobileNetV3LitePreprocessorQuantized(nn.Module):
    """
    Preprocessing module using MobileNetV3 Lite to convert image features to format suitable for KAN.
    Leverages pretrained MobileNetV3 Lite for efficient feature extraction.
    """
    def __init__(self, output_features=48,  
                 pretrained=True, freeze_mobilenet=True):
        super(MobileNetV3LitePreprocessorQuantized, self).__init__()
        
        # Load pretrained MobileNetV3 model
        # self.mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)        
        self.mobilenet = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        self.mobilenet.classifier = nn.Sequential()  # Rimuove il classificatore

        # Strato di dequantizzazione per l'interfaccia ibrida
        self.dequant = torch.quantization.DeQuantStub()

        # Final feature dimension after MobileNetV3
        mobilenet_output_dim = 576  # MobileNetV3 Lite final feature dimension
        
        # Linear projection to desired output feature size
        self.projector = nn.Sequential(
            nn.Linear(mobilenet_output_dim, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True)
        )

        # Quantizzazione dinamica
        #self.mobilenet = torch.quantization.quantize_dynamic(
        #    mobilenet_fp32, {torch.nn.Linear}, dtype=torch.qint8
        #)

        self.freeze_mobilenet = freeze_mobilenet
        self.is_qat_prepared = False

        # Freeze MobileNetV3 parameters if specified
        #if freeze_mobilenet:
        #    for param in self.mobilenet.parameters():
        #        param.requires_grad = False
        
        
    def forward(self, x):
        # MobileNetV2 expects 3 channels, but input might be 1 or 3.
        # If input_channels is 1, we need to convert to 3 channels.
        if x.size(1) == 1:
            # Convert grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        features = self.mobilenet(x)
        # features should be [batch, 1280]
        return self.projector(features)
    
    def prepare_for_qat(self):
        """Prepara MobileNetV3 per la Quantization-Aware Training (QAT) e la congela."""
        if self.is_qat_prepared:
            print("Il modello è già preparato per QAT.")
            return

        # 1. Fusione degli strati (Cruciale per la quantizzazione)
        # MobileNetV3 ha un metodo 'fuse_model'
        self.mobilenet.fuse_model()
        
        # 2. Imposta il qconfig QAT (per qint8)
        qconfig_qat = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 3. Assegna i qconfig: MobileNet per QAT, Proiettore escluso
        self.mobilenet.qconfig = qconfig_qat
        # Il qconfig è None per default per i moduli senza, ma lo impostiamo qui per chiarezza
        self.projector.qconfig = None 
        
        # 4. Applica la preparazione QAT al modulo completo
        torch.quantization.prepare_qat(self, inplace=True)
        self.is_qat_prepared = True

        # 5. Congelamento dei parametri QAT di MobileNetV3
        if self.freeze_mobilenet:
            self.freeze_mobilenet_parameters()
            
        print("Modello preparato per QAT ibrido. Pronto per l'addestramento.")
        
    def freeze_mobilenet_parameters(self):
        """Congela i pesi di MobileNetV3 per l'addestramento ibrido."""
        if self.is_qat_prepared:
            for name, param in self.mobilenet.named_parameters():
                param.requires_grad = False
            print("Parametri MobileNetV3 (QAT) congelati. Solo il Proiettore è addestrabile.")
        else:
            print("Prima devi chiamare prepare_for_qat().")

    def convert_to_quantized(self):
        """Converte il modello QAT addestrato nella versione finale quantizzata (qint8)."""
        if not self.is_qat_prepared:
             raise RuntimeError("Prima devi chiamare prepare_for_qat() e addestrare il modello.")

        # Imposta la modalità valutazione (necessaria per convert)
        self.eval() 
        
        # Converti il modello. Gli strati con qconfig=None rimangono FP32
        torch.quantization.convert(self, inplace=True)
        self.is_qat_prepared = False
        
        print("Conversione finale in MobileNetV3 (qint8) + Proiettore (FP32) completata.")

    def quantize_mobilenet_ptq(calibration_dataloader, pretrained=True):
        """
        Esegue la Quantizzazione Post-Addestramento (PTQ) su MobileNetV3 Small.
        """
        print("Inizio PTQ per MobileNetV3 Small...")
        
        # 1. Carica il modello FP32 base
        mobilenet_fp32 = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        mobilenet_fp32.classifier = nn.Sequential() # Rimuove il classificatore
        mobilenet_fp32.eval()
        
        # 2. Fusione e Preparazione
        mobilenet_fp32.fuse_model()
        qconfig = torch.quantization.get_default_qconfig('fbgemm') # qint8 per CPU
        mobilenet_fp32.qconfig = qconfig
        
        model_prepared = torch.quantization.prepare(mobilenet_fp32, inplace=False)
        
        # 3. Calibrazione
        print("Inizio Calibrazione...")
        with torch.no_grad():
            for inputs in calibration_dataloader:
                # Assicurati che l'input sia nel formato corretto [B, C, H, W]
                # Assumiamo che il dataloader fornisca il tensore di input direttamente o come primo elemento.
                if isinstance(inputs, (tuple, list)):
                    input_tensor = inputs[0] 
                else:
                    input_tensor = inputs
                _ = model_prepared(input_tensor)
        print("Calibrazione completata.")
    
        # 4. Conversione finale a qint8
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        print("PTQ completata. MobileNetV3 convertita in qint8.")
    
        return model_quantized

"""
class ImagePreprocessor(nn.Module):
    ""
    Preprocessing module to convert image features to format suitable for KAN.
    Uses lightweight convolutional layers to extract features while keeping parameters low.
    ""
    def __init__(self, input_channels=3, output_features=48, img_size=128, 
                 conv_channels=[16, 32, 64], kernel_size=3, pool_kernel_size=2,
                 final_pool_size=4, use_batch_norm=True, dropout_rate=0.1,
                 stochastic_depth_rate=0.0):
        super(ImagePreprocessor, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        num_blocks = len(conv_channels)
        
        # Create convolutional layers dynamically based on config
        for i, out_channels in enumerate(conv_channels):
            # Calculate current stochastic depth probability
            # Linearly increase probability from 0 -> stochastic_depth_rate
            curr_stochastic_depth_prob = stochastic_depth_rate * i / (num_blocks - 1) if num_blocks > 1 else 0
            
            conv_block = []
            # Convolution
            conv_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                        stride=1, padding=kernel_size//2))
            
            # Normalization
            if use_batch_norm:
                conv_block.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            conv_block.append(nn.ReLU(inplace=True))
            
            # Dropout for regularization
            if dropout_rate > 0:
                conv_block.append(nn.Dropout2d(dropout_rate))
            
            # Pooling
            conv_block.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))
            
            # Stochastic depth
            if stochastic_depth_rate > 0 and i < num_blocks - 1:  # No stochastic depth for last layer
                conv_block.append(StochasticDepth(curr_stochastic_depth_prob))
            
            self.conv_layers.append(nn.Sequential(*conv_block))
            in_channels = out_channels
        
        # Final pooling
        self.final_pool = nn.AdaptiveAvgPool2d((final_pool_size, final_pool_size))
        
        # Calculate feature size after convolutions and pooling
        self.feature_size = conv_channels[-1] * final_pool_size * final_pool_size
        
        # Linear projection to desired output feature size
        self.projection = nn.Sequential(
            nn.Linear(self.feature_size, output_features),
            nn.BatchNorm1d(output_features) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x):
        # Apply each convolutional block
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Apply final pooling
        x = self.final_pool(x)
        
        # Flatten and project
        x = x.view(-1, self.feature_size)
        x = self.projection(x)
        
        return x
"""

class RegularizedKAN(nn.Module):
    """
    A wrapper around the KAN class that adds regularization techniques
    like dropout and activation regularization.
    """
    def __init__(self, width, grid=4, degree=3, 
                 dropout_rate=0.0, use_batchnorm=False, activation_l1=0.0, seed=42):
        super(RegularizedKAN, self).__init__()
        
        self.kan = KAN(width=width, grid=grid, k=degree, seed=seed)
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.activation_l1 = activation_l1
        self.n_layers = len(width) - 1  # Number of KAN layers
        
        # Create batch norm layers if requested
        if use_batchnorm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(width[i+1]) for i in range(len(width)-2)
            ])
        
        # Store activations for regularization if needed
        self.activations = []
        
    def forward(self, x):
        # Reset activations
        self.activations = []
        
        # Forward pass through KAN with regularization
        # Store the input for activation regularization if needed
        if self.activation_l1 > 0 and self.training:
            self.activations.append(x)
        
        # Process through KAN
        x = self.kan(x)
        
        # Add output to activations
        if self.activation_l1 > 0 and self.training:
            self.activations.append(x)
        
        return x
    
    def get_activation_regularization(self):
        """Calculate L1 regularization on activations (sparsity)"""
        if self.activation_l1 <= 0 or not self.activations:
            return 0.0
        
        # Apply L1 regularization to all activations except output
        reg_loss = 0.0
        for act in self.activations[:-1]:  # Exclude output layer
            reg_loss += torch.mean(torch.abs(act)) * self.activation_l1
        
        return reg_loss
    
    def plot_groups(self, save_dir='figures'):
        """Visualize KAN splines and save to directory"""
        try:
            self.kan.plot_groups(save_dir=save_dir)
        except AttributeError:
            print("KAN implementation doesn't support plot_groups. Using available visualization...")
            try:
                self.kan.plot(save_dir=save_dir)
            except AttributeError:
                print("KAN visualization methods not available in this implementation.")


class KANImageClassifier(nn.Module):
    """
    KAN-based image classifier optimized for resource-constrained IoT devices.
    Combines convolutional feature extraction with KAN for efficient processing.
    Includes regularization techniques to prevent overfitting.
    """
    def __init__(self, input_channels=3, img_size=224, num_classes=2, feature_dim=48, 
                 kan_hidden_dims=[24, 12], kan_grid=4, kan_degree=3, conv_channels=[16, 32, 64],
                 use_batch_norm=True, dropout_rate=0.1, activation_l1=0.0, 
                 stochastic_depth_rate=0.0, seed=42, preprocessor_type='mobilenetv3_small', width_mult=1.0,
                 preprocessor_freeze=True, preprocessor_pretrained=True, head_type='kan'):
        super(KANImageClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.head_type = head_type

        # Image preprocessing network
        #self.preprocessor = ImagePreprocessor(
        #    input_channels=input_channels,
        #    output_features=feature_dim,
        #    img_size=img_size,
        #    conv_channels=conv_channels,
        #    use_batch_norm=use_batch_norm,
        #    dropout_rate=dropout_rate,
        #    stochastic_depth_rate=stochastic_depth_rate
        #)
        #         

        # Image preprocessing network
        # Image dimension is fixed to 224x224 for MobileNetV2
        # If input_channels is not 3, we handle it in forward
        # Structure is fixed due to pretrained model
        # Adaptation layer to project to feature_dim is needed to be trained
        
        if preprocessor_type == 'mobilenetv2':
            self.preprocessor = MobileNetV2Preprocessor(
            output_features=feature_dim,
            )
        elif preprocessor_type == 'mobilenetv3_small':
            self.preprocessor = MobileNetV3LitePreprocessor(
                output_features=feature_dim, width_mult=width_mult, 
                pretrained=preprocessor_pretrained, 
                freeze_mobilenet=preprocessor_pretrained,
                img_size=img_size
            )
        elif preprocessor_type == 'mobilenetv3_small_quantized':
            self.preprocessor = MobileNetV3LitePreprocessorQuantized(
            output_features=feature_dim,
            )
        elif preprocessor_type == 'kkan':
            self.preprocessor = KKAN_VWW_96(
            output_features=feature_dim,
            )

        if head_type == 'kan':
                # KAN network for classification with regularization
                kan_width = [feature_dim] + kan_hidden_dims + [num_classes]
                self.kan = RegularizedKAN(
                    width=kan_width,
                    grid=kan_grid,
                degree=kan_degree,
                dropout_rate=dropout_rate,
                use_batchnorm=use_batch_norm,
                activation_l1=activation_l1,
                seed=seed
            )
        elif head_type == 'mlp':
            # KAN hidden dims are used as mlp hidden dims
            self.kan = nn.Sequential(
                nn.Linear(feature_dim, kan_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(kan_hidden_dims[0], num_classes)
            )
        
    def forward(self, x):
        # Preprocess image to extract features
        features = self.preprocessor(x)
        
        # Process features through KAN
        output = self.kan(features)
        
        return output
    
    def get_model_size(self):
        """Calculate model size in MB"""
        # Total model size
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        # CNN preprocessor size
        cnn_size = 0
        for param in self.preprocessor.parameters():
            cnn_size += param.nelement() * param.element_size()
        
        # KAN size
        kan_size = 0
        for param in self.kan.parameters():
            kan_size += param.nelement() * param.element_size()
        
        total_mb = param_size / (1024 * 1024)
        cnn_mb = cnn_size / (1024 * 1024)
        kan_mb = kan_size / (1024 * 1024)
        
        return total_mb
    
    def get_parameter_count(self):
        """Return total and trainable parameter counts"""
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # CNN preprocessor parameters
        cnn_params = sum(p.numel() for p in self.preprocessor.parameters())
        cnn_trainable = sum(p.numel() for p in self.preprocessor.parameters() if p.requires_grad)
        
        # KAN parameters
        kan_params = sum(p.numel() for p in self.kan.parameters())
        kan_trainable = sum(p.numel() for p in self.kan.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'cnn': cnn_params,
            'kan': kan_params
        }
    
    def get_activation_regularization(self):
        """Get activation regularization from KAN"""
        if self.head_type == 'kan':
            return self.kan.get_activation_regularization()
        else:
            return None
    
    def visualize_splines(self, save_dir='figures'):
        """Visualize KAN splines and save to directory"""
        self.kan.plot_groups(save_dir=save_dir)