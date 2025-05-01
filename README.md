# Token-Shuffle: Towards High-Resolution Image Generation with Autoregressive Models

[![arXiv](https://img.shields.io/badge/arXiv-2504.17789-b31b1b.svg)](https://arxiv.org/abs/2504.17789)


## Example Usage

```python
# Hyperparameters
batch_size = 2
h, w = 16, 16  # Image feature map size
embed_dim = 768
shuffle_window_size = 2

# Create random input tokens
input_tokens = torch.randn(batch_size, h*w, embed_dim)  # Shape: [2, 256, 768]

# TokenShuffle
shuffled_tokens = token_shuffle(input_tokens)  # Shape: [2, 64, 768]

# TokenUnshuffle
unshuffled_tokens = token_unshuffle(shuffled_tokens)  # Shape: [2, 256, 768]
```

The TokenShuffle module reduces the sequence length by a factor of s² (where s is the shuffle_window_size) while preserving the essential information through spatial token fusion. In this example, the sequence length is reduced from 256 to 64, enabling more efficient processing of visual tokens.