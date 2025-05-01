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
                nn.Linear(transformer_dim, transformer_dim),
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
        x = self.input_compression_mlp(x)

        h = w = int(num_tokens ** 0.5)

        x = rearrange(x, 'b (h w) d -> b h w d', h=h, w=w)

        # Reshape to group local s×s windows
        # From [b, h, w, d] to [b, h//s, s, w//s, s, d]
        x = rearrange(x, 'b (h_new s1) (w_new s2) d -> b h_new s1 w_new s2 d', 
                     s1=s, s2=s)
        
        # Shuffle the dimensions to get [b, h//s, w//s, s, s, d]
        x = rearrange(x, 'b h_new s1 w_new s2 d -> b h_new w_new s1 s2 d')
        
        # Merge the local window dimensions (s×s) into embedding dimension
        # From [b, h//s, w//s, s, s, d] to [b, h//s, w//s, (s*s*d)]
        x = rearrange(x, 'b h_new w_new s1 s2 d -> b h_new w_new (s1 s2 d)')
        
        # Flatten spatial dimensions: [b, (h//s)*(w//s), embed_dim]
        x = rearrange(x, 'b h w d -> b (h w) d')


        # Apply feature fusion MLPs
        for mlp in self.feature_fusion_mlps:
            x = mlp(x)
        
        return x
    
    def token_unshuffle(self, x):
        """
        Expand merged tokens back to original token count
        
        Args:
            x (torch.Tensor): Merged tokens of shape [batch_size, num_merged_tokens, compressed_dim]
        
        Returns:
            torch.Tensor: Unshuffled tokens with original token count
        """
        batch_size, num_merged_tokens, dim = x.shape
        s = self.shuffle_window_size

        reduced_h, reduced_w = int(num_merged_tokens**0.5), int(num_merged_tokens**0.5)

        # Unshuffle the tokens
        x = rearrange(x, 'b (new_h new_w) (s1 s2 d) -> b new_h new_w s1 s2 d', s1=s, s2=s, new_h=reduced_h, new_w=reduced_w)

        x = rearrange(x, 'b new_h new_w s1 s2 d -> b new_h s1 new_w s2 d')

        x = rearrange(x, 'b new_h s1 new_w s2 d -> b (new_h s1) (new_w s2) d')

        x = rearrange(x, 'b h w d -> b (h w) d')

        # Restore original dimension
        x_restored = self.output_expansion_mlp(x)
        
        return x_restored
    
    def forward(self, x):
        """
        Forward pass through Token-Shuffle
        
        Args:
            x (torch.Tensor): Input tokens
        
        Returns:
            torch.Tensor: Processed tokens
        """
        # compress input tokens + shuffle
        x_shuffled = self.token_shuffle(x)

        # Process shuffled tokens (simulated Transformer computation)
        x_processed = x_shuffled  # Replace with actual Transformer processing
        
        # Unshuffle tokens + restore original dimension
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