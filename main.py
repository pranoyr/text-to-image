import torch
import torch.nn as nn
from einops import rearrange

class TokenShuffleLayer(nn.Module):
    def __init__(self, 
                 transformer_dim, 
                 shuffle_window_size=2, 
                 num_mlp_blocks=2):
        super().__init__()
        
        # Dimension compression factor
        self.shuffle_window_size = shuffle_window_size
        self.compressed_dim = transformer_dim // (shuffle_window_size ** 2)
        
        # Input compression MLP
        self.input_compression_mlp = nn.Sequential(
            nn.Linear(transformer_dim, self.compressed_dim),
            nn.GELU()
        )
        
        # Feature fusion MLP blocks
        self.feature_fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.compressed_dim, self.compressed_dim),
                nn.GELU()
            ) for _ in range(num_mlp_blocks)
        ])
        
        # Output expansion MLP
        self.output_expansion_mlp = nn.Sequential(
            nn.Linear(self.compressed_dim, transformer_dim),
            nn.GELU()
        )
    
    def token_shuffle(self, x):
        """
        Merge local tokens into fewer tokens
        
        Args:
            x (torch.Tensor): Input tokens of shape [batch_size, num_tokens, dim]
        
        Returns:
            torch.Tensor: Shuffled tokens with reduced token count
        """
        batch_size, num_tokens, dim = x.shape
        s = self.shuffle_window_size
        
        # Compress dimension
        x_compressed = self.input_compression_mlp(x)

        x_grouped = rearrange(x_compressed, 'b (t1 t2) d -> b t1 t2 d', t1=num_tokens // (s * s), t2=s * s)

        x_merged = x_grouped.mean(dim=2)
        
        # Apply feature fusion MLPs
        for mlp in self.feature_fusion_mlps:
            x_merged = mlp(x_merged)
        
        return x_merged
    
    def token_unshuffle(self, x):
        """
        Expand merged tokens back to original token count
        
        Args:
            x (torch.Tensor): Merged tokens of shape [batch_size, num_merged_tokens, compressed_dim]
        
        Returns:
            torch.Tensor: Unshuffled tokens with original token count
        """
        batch_size, num_merged_tokens, compressed_dim = x.shape
        s = self.shuffle_window_size

        # Expand tokens
        x_expanded = x.repeat_interleave(s * s, dim=1)

        # Restore original dimension
        x_restored = self.output_expansion_mlp(x_expanded)
        
        return x_restored
    
    def forward(self, x):
        """
        Forward pass through Token-Shuffle
        
        Args:
            x (torch.Tensor): Input tokens
        
        Returns:
            torch.Tensor: Processed tokens
        """
        # Shuffle tokens for Transformer computation
        x_shuffled = self.token_shuffle(x)
        
        # Process shuffled tokens (simulated Transformer computation)
        x_processed = x_shuffled  # Replace with actual Transformer processing
        
        # Unshuffle tokens back to original count
        x_unshuffled = self.token_unshuffle(x_processed)
        
        return x_unshuffled

# Example usage
def main():
    # Hyperparameters
    batch_size = 1
    num_tokens = 256
    transformer_dim = 768
    shuffle_window_size = 2
    
    # Create Token-Shuffle layer
    token_shuffle_layer = TokenShuffleLayer(
        transformer_dim=transformer_dim, 
        shuffle_window_size=shuffle_window_size
    )
    
    # Generate random input tokens
    input_tokens = torch.randn(
        batch_size, 
        num_tokens, 
        transformer_dim
    )
    
    # Apply Token-Shuffle
    output_tokens = token_shuffle_layer(input_tokens)
    
    print("Input tokens shape:", input_tokens.shape)
    print("Output tokens shape:", output_tokens.shape)

if __name__ == "__main__":
    main()