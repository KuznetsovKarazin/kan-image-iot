import torch
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
import config

# Get default model path
exp_paths = config.get_experiment_paths()
model_path = exp_paths['model_dir'] / 'kan_person_detector_best.pt'

print(f"Loading checkpoint: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("\n" + "="*60)
print("CHECKPOINT STRUCTURE")
print("="*60)
if isinstance(checkpoint, dict):
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    if 'model_info' in checkpoint:
        print(f"\nmodel_info keys: {list(checkpoint['model_info'].keys())}")
        if 'config' in checkpoint['model_info']:
            config_dict = checkpoint['model_info']['config']
            print(f"\nconfig keys: {list(config_dict.keys())}")
            if 'preprocessor' in config_dict:
                print(f"\nPreprocessor config: {config_dict['preprocessor']}")
            if 'kan' in config_dict:
                print(f"\nKAN config: {config_dict['kan']}")
    
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print(f"\nDirect config keys: {list(config_dict.keys())}")
        if 'preprocessor' in config_dict:
            print(f"\nPreprocessor config: {config_dict['preprocessor']}")
        if 'kan' in config_dict:
            print(f"\nKAN config: {config_dict['kan']}")
    
    # Check actual weight dimensions
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Check a known MobileNetV3 layer that changes with width_mult
        key_to_check = 'preprocessor.mobilenet.features.12.0.weight'
        if key_to_check in state_dict:
            shape = state_dict[key_to_check].shape
            print(f"\n" + "="*60)
            print("ACTUAL WEIGHT DIMENSIONS")
            print("="*60)
            print(f"Key: {key_to_check}")
            print(f"Shape: {shape}")
            
            # For MobileNetV3 small, features.12.0 is the final conv
            # wm=1.0: [576, 96, 1, 1]
            # wm=0.75: [432, 72, 1, 1]
            # wm=0.5: [288, 48, 1, 1]
            
            out_channels = shape[0]
            if out_channels == 576:
                print(f"\n=> Weights indicate width_mult = 1.0")
            elif out_channels == 432:
                print(f"\n=> Weights indicate width_mult = 0.75")
            elif out_channels == 288:
                print(f"\n=> Weights indicate width_mult = 0.5")
            else:
                print(f"\n=> Weights indicate unknown width_mult (out_channels={out_channels})")
