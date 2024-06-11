
import torch

def create_causal_mask(seq_len):
    """Creates a causal mask for a transformer decoder.
    ensures each token in the sequence can only attend to earlier positions
    fill `-inf` for the mask positions
    A += attention_mask
    A = F.softmax(A, dim=-1) and softmax of -inf is 0 which masks them
    """

    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


