# Token-Shuffle: Towards High-Resolution Image Generation with Autoregressive Models

[![arXiv](https://img.shields.io/badge/arXiv-2504.17789-b31b1b.svg)](https://arxiv.org/abs/2504.17789)


## Example Usage

```python
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
```

The TokenShuffle module reduces the sequence length by a factor of s² (where s is the shuffle_window_size) while preserving the essential information through spatial token fusion. In this example, the sequence length is reduced from 256 to 64, enabling more efficient processing of visual tokens.