import torch

def create_causal_attention_mask(sequence_type_array, prefix_full_attention=False, prefix_len=None, action_attend_next_state=False):
    """
    sequence_type_array: [False, True, False, True, False, True] for action first
    sequence_type_array: [True, False, True, False, True, False] for state first
    False: Action
    True: State

    Actions have a causal mask
    States have full attention over states


    A0   [[False,  True,  True,  True,  True,  True],
    S1    [ True, False,  True, False,  True, False],
    A1    [False, False, False,  True,  True,  True],
    S2    [ True, False,  True, False,  True, False],
    A2    [False, False, False, False, False,  True],
    S3    [ True, False,  True, False,  True, False]]
            A0    S1      A1    S2      A2    S3
    """
    seq_len = len(sequence_type_array)
    is_state = torch.tensor(sequence_type_array, dtype=torch.bool)
    
    # Start with a standard causal mask (tril) to prevent attending to future elements
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))



    # Apply the custom rules
    for i in range(seq_len):
        if is_state[i]:
            # If the current token is a state, it can attend to all other states.
            # This logic needs to override the causal mask for other states.
            # A state at index i can attend to any state j, regardless of j's position.
            state_indices = torch.where(is_state)[0]
            action_indices = torch.where(~is_state)[0]
            mask[i, state_indices] = True
            mask[i, action_indices] = False
        else:
            if action_attend_next_state:
                mask[i, min(seq_len-1, i+1)] = True
            # If the current token is an action, it can only attend to
            # past actions and past/current states (this is handled by the tril mask).
            pass # The initial causal mask is sufficient here.

    if prefix_full_attention:
        assert prefix_len is not None
        mask[:, :prefix_len] = True

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
