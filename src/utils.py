# deep learning libraries
import torch
import numpy as np

# other libraries
import os
import random

def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None

def clean_tokens(tokens):
    """
    Cleans the input tokens by removing unwanted characters and splitting them into a list. 
    It also merges tokens that are 's or n't with the previous token.
    Args:
        tokens (str): in the format '['token' 'token' 'token']'.
    Returns:
        list: A list of cleaned tokens.
    """
    if isinstance(tokens, str):
        # Remove unwanted characters and split tokens by commas
        tokens = tokens.replace("[", "").replace("]", "").replace("'", "").split(",")
        # Strip extra spaces and filter out empty strings
        tokens = [t.strip() for t in tokens if t.strip()]

        # Create a list to store the cleaned tokens
        cleaned_tokens = []
        
        # Regex to capture tokens that end in 's or n't
        for i, token in enumerate(tokens):
            # If the token is 's or n't and not the first word, merge with the previous token
            if token in ["n't", "'s"]:
                # Merge 'n't or 's with the previous token
                if cleaned_tokens:
                    cleaned_tokens[-1] += token
                continue
            else:
                cleaned_tokens.append(token)

        # Join the tokens into a single string
        return " ".join(cleaned_tokens)