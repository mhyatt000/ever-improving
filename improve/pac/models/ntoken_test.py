# Configuration variables
n_patch_latents = 9
use_hand_rgb = False
act_pred = True
fwd_pred = True
fwd_pred_hand = False
sequence_length = 10
batch_size = 19
hidden_size = 384
chunk_size = 10

# Initialize tokens dictionary with base tokens
tokens = {
    "lang": 1,
    "state": 1,
    "patch": n_patch_latents,
    "obs": 1,
}

# Add hand related tokens if use_hand_rgb is True
if use_hand_rgb:
    tokens["hand_patch"] = n_patch_latents
    tokens["hand_obs"] = 1
else:
    tokens["hand_patch"] = 0
    tokens["hand_obs"] = 0

# Calculate total number of tokens
n_tokens = sum(tokens.values())

# Add action prediction tokens if act_pred is True
if act_pred:
    tokens["act_pred"] = chunk_size
    act_query_token_start_i = n_tokens
    n_tokens += chunk_size

# Add forward prediction tokens if fwd_pred is True
if fwd_pred:
    obs_query_token_start_i = n_tokens
    n_tokens += tokens["patch"] + tokens["obs"]

    # Add forward prediction hand tokens if fwd_pred_hand is True
    if fwd_pred_hand:
        obs_hand_query_token_start_i = n_tokens
        n_tokens += tokens["patch"] + tokens["obs"]

print(n_tokens)
quit()

# Layer norm
stacked_inputs = stacked_inputs.reshape(
    batch_size, n_tokens * sequence_length, hidden_size
)
stacked_inputs = embed_ln(stacked_inputs)
