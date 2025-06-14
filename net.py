import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network with policy and value heads"""
    
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        
        # Initial convolutional block
        self.initial_conv = nn.Conv2d(config.input_channels, config.num_filters, 
                                    kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(config.num_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters) for _ in range(config.residual_blocks_num)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(config.num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(config.N * config.N * 2, config.all_moves_num)
        
        # Value head
        self.value_conv = nn.Conv2d(config.num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.N * config.N, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        # Initial convolutional block
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)  # Raw logits (no softmax here)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ResidualBlock(nn.Module):
    """Residual block for deep convolutional network"""
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


if __name__ == '__main__':
    # Test the network
    print(f"PyTorch device available: {torch.cuda.is_available()}")
    
    model = AlphaZeroNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.input_channels, config.N, config.N)
    
    model.eval()
    with torch.no_grad():
        policy_pred, value_pred = model(dummy_input)
        print(f"Policy output shape: {policy_pred.shape}")
        print(f"Value output shape: {value_pred.shape}")