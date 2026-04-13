import torch
import torch.nn as nn

class OldFastLutKAN:
    def __init__(self, lut_table, min_val, max_val, grid_size, input_dim):
        self.lut_table = lut_table
        self.min_val = min_val
        self.max_val = max_val
        self.grid_size = grid_size
        self.input_dim = input_dim

    def forward(self, x):
        grid_step = (self.max_val - self.min_val) / (self.grid_size - 1)
        x_grid = ((x - self.min_val) / grid_step).unsqueeze(-1)
        grid = torch.arange(self.grid_size, device=x.device).float().view(1, 1, -1)
        dist = torch.abs(x_grid - grid)
        weight_mask = torch.relu(1.0 - dist)
        mask_permuted = weight_mask.permute(0, 2, 1)
        combined = mask_permuted.unsqueeze(-1) * self.lut_table.unsqueeze(0)
        res = combined.sum(dim=1)
        return res.mean(dim=1)

class NewFastLutKAN:
    def __init__(self, lut_table, min_val, max_val, grid_size, input_dim):
        self.lut_table = lut_table
        self.min_val = min_val
        self.max_val = max_val
        self.grid_size = grid_size
        self.input_dim = input_dim

    def forward(self, x):
        grid_step = (self.max_val - self.min_val) / (self.grid_size - 1)
        grid = torch.arange(self.grid_size, device=x.device).float()
        
        B = x.shape[0]
        res = torch.zeros(B, self.lut_table.shape[-1], device=x.device)
        
        for i in range(self.input_dim):
            x_i = x[:, i]
            x_grid_i = (x_i - self.min_val) / grid_step
            dist_i = torch.abs(x_grid_i.unsqueeze(1) - grid.unsqueeze(0))
            weight_i = torch.relu(1.0 - dist_i)
            lut_i = self.lut_table[:, i, :]
            res += torch.matmul(weight_i, lut_i)
            
        return res / self.input_dim

# Test parameters
B = 2
input_dim = 16
out_dim = 2
grid_size = 256
min_val = -4.0
max_val = 4.0

torch.manual_seed(42)
lut_table = torch.randn(grid_size, input_dim, out_dim)
x = torch.randn(B, input_dim) * 2

old_model = OldFastLutKAN(lut_table, min_val, max_val, grid_size, input_dim)
new_model = NewFastLutKAN(lut_table, min_val, max_val, grid_size, input_dim)

y_old = old_model.forward(x)
y_new = new_model.forward(x)

diff = torch.abs(y_old - y_new).max().item()
print(f"Max diff: {diff}")
print("Old output:")
print(y_old)
print("New output:")
print(y_new)
