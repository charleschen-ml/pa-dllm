import torch
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

def extract_numerical(text):
    """Extract the final boxed numerical answer (e.g., from \boxed{72}) or trailing number."""
    # Try to extract from \boxed{...}
    boxed_match = re.search(r"\\boxed{([\d.,]+)}", text)
    if boxed_match:
        num_str = boxed_match.group(1).replace(",", "")
    else:
        # Try to extract last number in string
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*$", text.strip())
        num_str = num_match.group(1) if num_match else None

    if num_str is None:
        return None
    try:
        return int(num_str) if '.' not in num_str else float(num_str)
    except ValueError:
        return None

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True) # count total of masked tokens

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate_vanilla(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                     cfg_scale=0., remasking='low_confidence', mask_id=126336, expected_answer=None):
    '''
    Args:
        model: Mask predictor.
        tokenizer: Tokenizer for decoding outputs.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        expected_answer: Optional, if set will compare against `extract_numerical()` for correctness logging.
    '''

    # Create x = prompt + completion 
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()  # initialize prompt, while leaving completion tokens as <mask>

    prompt_index = (x != mask_id)  # create boolean mask with prompt = T, completion = F
                                   # e.g. [T, T, T, ..., F, F, F, ...]
                                   # used later if cfg enabled

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks  # convert total_steps to steps_per_block

    first_correct_step = None  # Track first step with correct answer

    for num_block in range(num_blocks):

        # initialize boolean mask to all <mask> in current block
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: 
                                 prompt.shape[1] + (num_block + 1) * block_length] == mask_id)

        # calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            total_step = num_block * steps_per_block + i + 1  # total steps as efficiency metric

            mask_index = (x == mask_id)  # update the boolean mask (since last step)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits  # get logits with current x

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # get index of token with highest logit at each position

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)  # convert logits to probs

                # extract prob at each position with highest logit
                x0_p = torch.squeeze( 
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # mask out tokens beyond the current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # torch.where(mask, tensor_A, tensor_B): if mask_index is True, use tensor A, otherwise use tensor B
            # if token is true (masked), use x0 (token index with highest logit)
            # otherwise use x (original token)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):  # loop through each batch
                # torch.topk(input, k): selects the top k tokens from "input" (list)
                # returns (values, indices)
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])

                # use "advanced indexing" to set all indices in select_index
                # equivalent to saying:
                # for index in select_index:
                #   transfer_index[j, index] = True
                transfer_index[j, select_index] = True

            # unmask (freeze) the tokens in x (also using advanced indexing)
            x[transfer_index] = x0[transfer_index]

            # print intermediate outputs
            # out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            # print("\n" + out_text)
        
    # print final output
    # out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    # print("\n" + out_text)

    return x

@torch.no_grad()
def generate_charles(model, tokenizer, prompt, scheduler=None, steps=128, gen_length=128, block_length=128, 
                     temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, 
                     expected_answer=None, use_regression=True):
    '''
    Dynamic block size generation using XGBoost scheduler.
    
    Args:
        model: Mask predictor.
        tokenizer: Tokenizer for decoding outputs.
        prompt: A tensor of shape (1, L).
        scheduler: Trained XGBoost model for predicting block sizes. If None, uses fixed block_length.
        steps: Sampling steps (not used in dynamic version, kept for compatibility).
        gen_length: Generated answer length.
        block_length: Default block length if scheduler is None.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        expected_answer: Optional, if set will compare against `extract_numerical()` for correctness logging.
        use_regression: If True, scheduler is regression model; else classification.
    '''

    # Create x = prompt + completion 
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()  # initialize prompt, while leaving completion tokens as <mask>
    
    # Debug: Show prompt information
    prompt_text = tokenizer.batch_decode(prompt, skip_special_tokens=True)[0]
    print(f"\n{'='*80}")
    print(f"🎬 STARTING GENERATION")
    print(f"{'='*80}")
    print(f"📝 Prompt length: {prompt.shape[1]} tokens")
    print(f"📝 Prompt token IDs (first 20): {prompt[0, :20].tolist()}")
    print(f"📝 Prompt token IDs (last 20): {prompt[0, -20:].tolist()}")
    print(f"📝 Prompt text:\n{prompt_text}")
    print(f"\n🎯 Generation settings:")
    print(f"   - gen_length: {gen_length} tokens")
    print(f"   - mask_id: {mask_id}")
    print(f"   - Scheduler: {'XGBoost' if scheduler is not None else 'Fixed block size'}")
    print(f"{'='*80}\n")

    prompt_index = (x != mask_id)  # create boolean mask with prompt = T, completion = F
                                   # e.g. [T, T, T, ..., F, F, F, ...]
                                   # used later if cfg enabled

    # Track current position and features for XGBoost
    curr_pos = 0

    while curr_pos < gen_length:  # loop until we reach gen_length
        # Get current generation region
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length

        # print the size of x and decoded x
        print(f"The size of x: {x.shape}")
        decoded_x = tokenizer.batch_decode(x, skip_special_tokens=False)[0]
        print(f"The decoded x: {decoded_x}")

        # Get logits for current state
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits  # get logits with current x
        
        # Generate tokens for current block (semi-AR: unmask next N tokens left-to-right)
        # Use the predicted tokens with added noise for sampling
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        block_size = 1 # hardcode for debugging
        
        # Semi-autoregressive approach: Unmask the NEXT block_size tokens sequentially
        # (not top-confidence tokens, but next N positions from left to right)
        block_abs_start = gen_start + curr_pos
        block_abs_end = block_abs_start + block_size
        
        # Debug: Show mask state before unmasking
        # mask_index = (x == mask_id)
        
        # Replace only the next block_size positions with predicted tokens
        # torch.where(mask, tensor_A, tensor_B): if mask is True, use A, otherwise use B
        # x0_masked = torch.where(mask_index, x0, x)  # Use predictions only for masked positions
        x[:, block_abs_start:block_abs_end] = x0[:, block_abs_start:block_abs_end]  # Unmask next block
        
        # Move to next block
        curr_pos += block_size

        # Print intermediate outputs for debugging
        out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"\n{'='*80}")
        print(f"🔄 Step: Position {curr_pos-block_size} → {curr_pos}/{gen_length}")
        print(f"📊 Predicted block_size: {block_size} tokens")
        print(f"📝 Current generation:\n{out_text}")
        
        # Show what tokens were just unmasked
        newly_unmasked = x[:, block_abs_start:block_abs_end]
        newly_unmasked_text = tokenizer.batch_decode(newly_unmasked, skip_special_tokens=True)[0]
        print(f"🆕 Newly unmasked tokens: '{newly_unmasked_text}'")
        
    # print final output
    # out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    # print("\n" + out_text)

    return x

@ torch.no_grad()
def generate_custom(model, tokenizer, prompt, steps=128, gen_length=128, block_sizes=None, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, curr_pos=0, correct_answer=None, verbose=True):
    '''
    Args:
        model: Mask predictor.
        tokenizer: Tokenizer for decoding.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_sizes: List of block sizes. If None, uses uniform blocks of size gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        curr_pos: position where we want to save confidence and entropy.
        correct_answer: Expected correct answer for checking correctness (optional).
    '''
    # debug
    # print("\nstart generate_custom:\n")

    # Create x = prompt + completion 
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone() # initialize prompt, while leaving completion tokens as <mask>

    prompt_index = (x != mask_id) # create boolean mask with prompt = T, completion = F
                                # e.g. [T, T, T, ..., F, F, F, ...]
                                # used later if cfg enabled

    # Handle block sizes
    if block_sizes is None:
        # Default: single block of size gen_length
        block_sizes = [gen_length]
    
    # Validate block sizes
    total_block_size = sum(block_sizes)
    if total_block_size != gen_length:
        raise ValueError(f"Sum of block_sizes ({total_block_size}) must equal gen_length ({gen_length})")
    
    num_blocks = len(block_sizes)
    
    # Calculate steps per block (assume uniform distribution)
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # Define helper functions
    def _entropy(v: float) -> float:
        return round(-float(v) * float(np.log(max(v, 1e-12))), 4)
    
    def _shannon_entropy(probs: torch.Tensor) -> float:
        """Calculate Shannon entropy over full probability distribution"""
        # Ensure probabilities are positive and normalized
        probs = torch.clamp(probs, min=1e-12)
        probs = probs / probs.sum()  # Normalize just in case
        entropy = -torch.sum(probs * torch.log(probs))
        return round(float(entropy), 4)

    first_correct_step = None  # Track first step with correct answer
    block_confidences = {}  # Track confidence for each block
    # Track per-step confidences (decoded vs remaining) to print at the end
    per_step_logs = []
    
    # Initialize confidence and entropy - will be captured at curr_pos block
    initial_entropy = None
    initial_confidence = None
    
    # Track top-1 tokens after AR context is built (for parallel vs sequential analysis)
    ar_context_tokens = None
    additional_features = {}  # Store additional confidence metrics
    
    if cfg_scale > 0.:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        logits = model(x_).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x).logits
    
    # Calculate cumulative block positions
    block_starts = [0] + [sum(block_sizes[:i]) for i in range(1, len(block_sizes))]
    
    for num_block in range(num_blocks):
        block_size = block_sizes[num_block]
        block_start = block_starts[num_block]
        block_end = block_start + block_size

        # Skip blocks with zero size
        if block_size == 0:
            continue
            
        # Log block progression for verification
        # if num_block <= curr_pos + 1:  # Only log around curr_pos to avoid spam
        #     print(f"\n🔄 Processing block {num_block} (size: {block_size}, range: {block_start}-{block_end})")
        #     if num_block == curr_pos:
        #         print(f"   ⭐ This is the TARGET block (curr_pos={curr_pos}) - will capture confidence/entropy here!")

        # Capture confidence and entropy at curr_pos block (before processing steps)
        if num_block == curr_pos and initial_confidence is None:
            if verbose:
                print(f"\n🎯 CAPTURING confidence/entropy at block {num_block} (curr_pos={curr_pos})")
            
            # Get current logits to calculate confidence/entropy
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            # Show current state of generation
            current_decoded = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=True)
            if verbose:
                print(f"📝 Current generation state: '{current_decoded}'")
            
            # Count how many tokens are still masked
            gen_start = prompt.shape[1]
            gen_end = gen_start + gen_length
            masked_count = (x[0, gen_start:gen_end] == mask_id).sum().item()
            decoded_count = gen_length - masked_count
            if verbose:
                print(f"🎭 Tokens decoded so far: {decoded_count}/{gen_length} (masked: {masked_count})")
            
            # CAPTURE TOP-1 TOKENS after AR context is built (for parallel vs sequential analysis)
            if verbose:
                print(f"🔍 CAPTURING top-1 tokens after AR context (blocks 0 to {curr_pos-1} processed)")
            top1_tokens = torch.argmax(logits, dim=-1)  # Get top-1 predictions for all positions
            ar_context_tokens = x.clone()  # Start with current state (AR context + masks)
            ar_context_tokens[0, gen_start:gen_end] = top1_tokens[0, gen_start:gen_end]  # Fill with top-1 predictions
            
            ar_tokens_text = tokenizer.decode(ar_context_tokens[0, prompt.shape[1]:], skip_special_tokens=True)
            if verbose:
                print(f"📝 AR context + top-1 predictions: '{ar_tokens_text}'")
            
            # Calculate comprehensive confidence metrics
            p = F.softmax(logits, dim=-1)  # Full softmax probabilities
            gen_logits = logits[0, gen_start:gen_end]  # Logits for generation positions only
            gen_probs = p[0, gen_start:gen_end]  # Probabilities for generation positions only
            
            # Get top-k predictions for margin calculations
            top_probs, top_indices = torch.topk(gen_probs, k=min(2, gen_probs.shape[1]), dim=1)
            
            # Basic confidence and entropy (original)
            initial_conf = torch.squeeze(
                torch.gather(gen_probs, dim=-1, index=torch.unsqueeze(torch.argmax(gen_logits, dim=-1), -1)), -1
            )
            
            initial_confidence = [round(float(initial_conf[i]), 4) for i in range(gen_length)]
            initial_entropy = [_entropy(float(initial_conf[i])) for i in range(gen_length)]
            initial_shannon_entropy = [_shannon_entropy(gen_probs[i]) for i in range(gen_length)]
            
            # NEW FEATURES: Calculate additional metrics
            additional_features = {}
            
            for pos in range(gen_length):
                pos_probs = gen_probs[pos]  # Probabilities for this position
                pos_logits = gen_logits[pos]  # Logits for this position
                
                # Basic features
                conf_0 = float(initial_conf[pos])  # Confidence of next token
                entropy_0 = initial_entropy[pos]  # Entropy of next token (self-information)
                shannon_entropy_0 = initial_shannon_entropy[pos]  # Shannon entropy of full distribution
                
                # Top1 margin (difference between top-1 and top-2)
                pos_top_probs, _ = torch.topk(pos_probs, k=min(2, pos_probs.shape[0]))
                if len(pos_top_probs) >= 2:
                    top1_margin = float(pos_top_probs[0] - pos_top_probs[1])
                else:
                    top1_margin = float(pos_top_probs[0])  # Only one token available
                
                # Global features (mean/std across remaining tokens from current position)
                remaining_conf = initial_confidence[pos:]
                remaining_entropy = initial_entropy[pos:]
                remaining_shannon_entropy = initial_shannon_entropy[pos:]
                
                mean_confidence = float(np.mean(remaining_conf)) if remaining_conf else 0.0
                mean_entropy = float(np.mean(remaining_entropy)) if remaining_entropy else 0.0
                shannon_mean_entropy = float(np.mean(remaining_shannon_entropy)) if remaining_shannon_entropy else 0.0
                conf_std = float(np.std(remaining_conf)) if len(remaining_conf) > 1 else 0.0
                entropy_std = float(np.std(remaining_entropy)) if len(remaining_entropy) > 1 else 0.0
                shannon_entropy_std = float(np.std(remaining_shannon_entropy)) if len(remaining_shannon_entropy) > 1 else 0.0
                
                # Specific position features
                conf_1 = remaining_conf[1] if len(remaining_conf) > 1 else 0.0  # 2nd token confidence
                
                # Top-K vs Sequential features
                # Get all remaining confidences and find top-k vs next-k
                remaining_tensor = torch.tensor(remaining_conf)
                
                # Top4/8 confidence minimums (best 4/8 tokens regardless of position)
                top4_values, _ = torch.topk(remaining_tensor, k=min(4, len(remaining_tensor)), largest=True)
                top8_values, _ = torch.topk(remaining_tensor, k=min(8, len(remaining_tensor)), largest=True)
                top4_conf_min = float(top4_values[-1]) if len(top4_values) > 0 else 0.0  # Minimum of top 4
                top8_conf_min = float(top8_values[-1]) if len(top8_values) > 0 else 0.0  # Minimum of top 8
                
                # Next4/8 confidence minimums (immediate next 4/8 sequential tokens)
                next4_conf = remaining_conf[:4]  # Next 4 sequential tokens
                next8_conf = remaining_conf[:8]  # Next 8 sequential tokens
                next4_conf_min = float(min(next4_conf)) if next4_conf else 0.0
                next8_conf_min = float(min(next8_conf)) if next8_conf else 0.0
                
                # Store all features for this position
                additional_features[pos] = {
                    'conf_0': conf_0,
                    'entropy_0': entropy_0,
                    'shannon_entropy_0': shannon_entropy_0,
                    'top1_margin': top1_margin,
                    'mean_confidence': mean_confidence,
                    'mean_entropy': mean_entropy,
                    'shannon_mean_entropy': shannon_mean_entropy,
                    'conf_std': conf_std,
                    'entropy_std': entropy_std,
                    'shannon_entropy_std': shannon_entropy_std,
                    'conf_1': conf_1,
                    'top4_conf_min': top4_conf_min,
                    'next4_conf_min': next4_conf_min,
                    'top8_conf_min': top8_conf_min,
                    'next8_conf_min': next8_conf_min,
                }
            
            # Show sample of captured values
            if verbose:
                print(f"📊 Sample confidence values: {initial_confidence[:5]}...")
                print(f"📈 Sample entropy values: {initial_entropy[:5]}...")
                print(f"✅ Captured {len(initial_confidence)} confidence/entropy pairs")
                print("=" * 60)

        # initialize boolean mask to all <mask> in current block
        block_mask_index = (x[:, prompt.shape[1] + block_start: prompt.shape[1] + block_end] == mask_id)

        # calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            total_step = num_block * steps_per_block + i + 1 # total steps as efficiency metric

            
            mask_index = (x == mask_id) # update the boolean mask (since last step)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits # get logits with current x

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # get index of token with highest logit at each position

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1) # convert logits to probs
                
                # extract prob at each position with highest logit
                x0_p = torch.squeeze( 
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Also compute full-range top confidences before masking to block (for logging)
            x0_p_full = torch.squeeze(
                torch.gather(F.softmax(logits, dim=-1), dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )  # shape: (b, seq_len)

            # mask out tokens beyond the current block (for sampling decision only)
            x0_p[:, prompt.shape[1] + block_end:] = -np.inf

            # torch.where(mask, tensor_A, tensor_B): if mask_index is True, use tensor A, otherwise use tensor B
            # if token is true (masked), use x0 (token index with highest logit)
            # otherwise use x (original token)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]): # loop through each batch
                # torch.topk(input, k): selects the top k tokens from "input" (list)
                # returns (values, indices)
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                
                # use "advanced indexing" to set all indices in select_index
                # equivalent to saying:
                # for index in select_index:
                #   transfer_index[j, index] = True
                transfer_index[j, select_index] = True

            # For logging: split full confidences into decoded-this-step vs remaining-masked
            gen_start = prompt.shape[1]
            gen_end = gen_start + gen_length
            all_conf = x0_p_full[0, gen_start:gen_end]

            # Indices within generation region that will be decoded this step
            dec_mask_gen = transfer_index[0, gen_start:gen_end]
            decoded_indices = torch.nonzero(dec_mask_gen, as_tuple=False).flatten()

            # Remaining masked positions this step (currently masked AND not selected to decode now)
            masked_gen = mask_index[0, gen_start:gen_end]
            remaining_mask_indices = torch.nonzero(masked_gen & (~dec_mask_gen), as_tuple=False).flatten()

            decoded_conf = [round(float(all_conf[idx]), 4) for idx in decoded_indices]
            remaining_conf = [round(float(all_conf[idx]), 4) for idx in remaining_mask_indices]

            # Entropy values for decoded vs remaining

            decoded_entropy = [_entropy(float(all_conf[idx])) for idx in decoded_indices]
            remaining_entropy = [_entropy(float(all_conf[idx])) for idx in remaining_mask_indices]
            
            # Calculate Shannon entropy for decoded and remaining positions
            decoded_shannon_entropy = []
            remaining_shannon_entropy = []
            
            for idx in decoded_indices:
                # Get full probability distribution for this position
                pos_probs = p[0, idx]  # p is the full softmax probabilities
                decoded_shannon_entropy.append(_shannon_entropy(pos_probs))
            
            for idx in remaining_mask_indices:
                # Get full probability distribution for this position
                pos_probs = p[0, idx]  # p is the full softmax probabilities
                remaining_shannon_entropy.append(_shannon_entropy(pos_probs))

            per_step_logs.append({
                'step': int(total_step),
                'block': int(num_block),
                'decoded_pos': decoded_indices.detach().cpu().tolist(),
                'remaining_pos': remaining_mask_indices.detach().cpu().tolist(),
                'decoded_conf': decoded_conf,
                'remaining_conf': remaining_conf,
                'decoded_entropy': decoded_entropy,
                'remaining_entropy': remaining_entropy,
                'decoded_shannon_entropy': decoded_shannon_entropy,
                'remaining_shannon_entropy': remaining_shannon_entropy,
            })

            # unmask (freeze) the tokens in x (also using advanced indexing)
            x[transfer_index] = x0[transfer_index]
            
            # Store confidence for this block if this is the last step of the block
            if i == steps_per_block - 1:  # Last step of this block

                block_confidence = []
                for j in range(block_size):
                    token_pos = prompt.shape[1] + block_start + j
                    if token_pos < confidence.shape[1]:
                        conf_val = confidence[0, token_pos].item()

                        if conf_val != -np.inf:  # Only include non-masked tokens
                            block_confidence.append(conf_val)

                if block_confidence:
                    block_confidences[num_block] = block_confidence


            # check answer correct
            out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            # print("\n" + out_text)
            is_correct = False
            if correct_answer is not None:
                extracted_answer = extract_numerical(out_text)
                is_correct = extracted_answer == correct_answer
                if is_correct and first_correct_step is None:
                    first_correct_step = total_step
            # print(f"{'✅' if is_correct else '❌'} | step: {total_step}")

    if verbose:
        print(f"\nFirst correct answer found at step: {first_correct_step if first_correct_step is not None else float('inf')}")

    # Print per-step confidence breakdown at the end
    if per_step_logs and verbose:
        print(f"\n{'='*60}")
        print("PER-STEP CONFIDENCE BREAKDOWN (decoded | remaining)")
        print(f"{'='*60}")
        for log in per_step_logs:
            print(f"step {log['step']} (block {log['block']}):")
            print(f"  top confidence: {log['decoded_conf']} {log['remaining_conf']}")
            print(f"  entropy: {log['decoded_entropy']} {log['remaining_entropy']}")
            if 'decoded_shannon_entropy' in log:
                print(f"  shannon entropy: {log['decoded_shannon_entropy']} {log['remaining_shannon_entropy']}")
        print(f"{'='*60}")

    # block_confidences: Final confidence scores for tokens that were actually decoded in each block
    return x, first_correct_step if first_correct_step is not None else float('inf'), block_confidences, initial_entropy, initial_confidence, ar_context_tokens, additional_features, initial_shannon_entropy


@torch.no_grad()
def generate_block_incremental(model, x, block_start, block_size, 
                                temperature=0., cfg_scale=0., remasking='low_confidence', 
                                mask_id=126336, prompt_length=0, return_features=False):
    """
    Generate a single block incrementally without regenerating previous tokens.
    
    KEY DIFFERENCE VS generate_vanilla:
    - generate_vanilla: Creates x from scratch, generates ALL blocks in one call
    - generate_block_incremental: Takes EXISTING x, generates ONLY ONE block
    
    EFFICIENCY:
    - Called once per block (e.g., 5 times for [1, 7, 8, 8, 8])
    - Each call only processes ONE block, not all 32 tokens
    - Previously generated tokens are PRESERVED, not regenerated
    - Hard-coded to steps_per_block=1 (decode all tokens in one shot, no iterative refinement)
    
    Args:
        model: Mask predictor model
        x: Current state [1, prompt_length + gen_length] with some tokens already unmasked
           CRITICAL: This is the SHARED state across all blocks!
           - On 1st call: x has prompt + 32 masked tokens
           - On 2nd call: x has prompt + N unmasked + (32-N) masked tokens (from previous call)
           - On 3rd call: x has prompt + M unmasked + (32-M) masked tokens (from previous calls)
           - etc.
        block_start: Starting position of this block (relative to completion, not prompt)
                     e.g., 0, 3, 8, 16, 24 for schedule [3, 5, 8, 8, 8]
        block_size: Number of tokens to generate in this block (determined by XGBoost)
                    e.g., 3, 5, 8, 8, 8 for schedule [3, 5, 8, 8, 8]
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: 'low_confidence' or 'random'
        mask_id: Token ID for [MASK] (126336)
        prompt_length: Length of the prompt (to know where completion starts)
        return_features: If True, return confidence/entropy features
        
    Returns:
        x: Updated tensor with this block generated (MODIFIED IN-PLACE via torch.where)
        features: Dict with confidence/entropy if return_features=True
    """
    prompt_index = torch.arange(prompt_length, device=x.device) if cfg_scale > 0 else None
    
    # ============================================================================
    # SETUP: Define block boundaries in absolute tensor indexing
    # ============================================================================
    # Example: If prompt_length=100, block_start=8, block_size=8
    #   block_abs_start = 108 (start of this block in full tensor)
    #   block_abs_end = 116 (end of this block in full tensor)
    block_abs_start = prompt_length + block_start
    block_abs_end = prompt_length + block_start + block_size
    
    # DIFFERENCE: generate_vanilla calculates this inside a loop over ALL blocks.
    # Here, we only care about ONE block (the current one).
    
    # ============================================================================
    # EFFICIENCY KEY #1: Only mask for CURRENT block
    # ============================================================================
    # Check which tokens in THIS BLOCK are still masked
    # Previously generated blocks (positions < block_start) are already unmasked!
    block_mask_index = (x[:, block_abs_start:block_abs_end] == mask_id)
    
    # DIFFERENCE: generate_vanilla processes block_mask_index for each block sequentially.
    # Here, blocks before block_start are ALREADY done (unmasked in previous calls).
    
    # ============================================================================
    # SCHEDULER-GUIDED APPROACH: Unmask all tokens in ONE SHOT (steps_per_block=1)
    # ============================================================================
    # Since XGBoost already determined we should generate block_size tokens,
    # we decode ALL of them in one forward pass (no iterative refinement)
    #
    # DIFFERENCE from generate_vanilla's confidence-based unmasking:
    # - generate_vanilla: Uses get_num_transfer_tokens() to gradually unmask by confidence
    # - generate_block_incremental: XGBoost decided block_size, decode all at once
    #
    # Hard-coded: steps_per_block=1 (decode all tokens in block in one shot)
    steps_per_block = 1
    num_transfer_tokens = block_size  # Unmask all block_size tokens in one step
    
    # Storage for features (if requested)
    confidences = [] if return_features else None
    entropies = [] if return_features else None
    
    # ============================================================================
    # SINGLE-SHOT DECODING: Unmask all tokens in this block at once
    # ============================================================================
    # Only one iteration since steps_per_block=1
    for i in range(steps_per_block):
        # ========================================================================
        # EFFICIENCY KEY #2: mask_index checks ALL tokens, but previously 
        # unmasked tokens will stay unmasked due to torch.where() below
        # ========================================================================
        mask_index = (x == mask_id)  # Boolean tensor: True = still masked
        
        # SAME AS generate_vanilla: Get logits from model
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[:, :prompt_length] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits  # Forward pass through model
        
        # SAME AS generate_vanilla: Get predicted tokens
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # Predicted tokens at each position
        
        # SAME AS generate_vanilla: Get confidence scores for remasking
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
        
        # ========================================================================
        # EFFICIENCY KEY #3: Mask out tokens BEYOND current block
        # ========================================================================
        # Set confidence to -inf for positions >= block_abs_end
        # This ensures we ONLY unmask tokens in the CURRENT block
        x0_p[:, block_abs_end:] = -np.inf
        
        # DIFFERENCE: generate_vanilla uses:
        #   x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
        # which masks out future blocks. Here we do the same but for only ONE block.
        
        # ========================================================================
        # EFFICIENCY KEY #4: Preserve already-unmasked tokens
        # ========================================================================
        # x0 = torch.where(mask_index, x0, x)
        # Translation: For each position:
        #   - If mask_index[pos] == True (still masked): use x0[pos] (new prediction)
        #   - If mask_index[pos] == False (already unmasked): use x[pos] (keep old value)
        # 
        # This is HOW we avoid regeneration!
        # Tokens from previous blocks are already unmasked, so they get preserved here.
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
        
        # ========================================================================
        # HYBRID APPROACH: XGBoost determines HOW MANY, confidence determines WHICH
        # ========================================================================
        # - XGBoost decided block_size (e.g., 7 tokens) → decode all 7 at once
        # - Confidence decides WHICH of those 7 tokens to unmask (highest confidence first)
        # This combines XGBoost's high-level scheduling with the model's token-level confidence
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            # Since steps_per_block=1, we unmask all num_transfer_tokens in this one step
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens)
            transfer_index[j, select_index] = True
        
        # ========================================================================
        # EFFICIENCY KEY #5: Unmask selected tokens IN-PLACE (modifies x directly)
        # ========================================================================
        # This MODIFIES the input tensor x!
        # After this line, x contains newly unmasked tokens from this refinement step
        x[transfer_index] = x0[transfer_index]
        
        # SAME AS generate_vanilla, but crucial difference:
        # In generate_vanilla, x is local to the function
        # Here, x is SHARED across all calls to generate_block_incremental!
        # So this modification PERSISTS to the next call.
        
        # Store features if requested
        if return_features:
            # Get confidence/entropy for the first token in this block
            if i == 0 and block_start < x.shape[1] - prompt_length:
                token_pos = block_abs_start
                conf = x0_p[0, token_pos].item()
                # Calculate entropy
                probs = F.softmax(logits[0, token_pos], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                confidences.append(conf)
                entropies.append(entropy)
    
    # ============================================================================
    # RETURN: x is modified in-place, so caller gets updated state
    # ============================================================================
    if return_features:
        return x, {'confidences': confidences, 'entropies': entropies}
    return x


# ============================================================================
# VISUAL EXAMPLE: How generate_block_incremental avoids regeneration
# ============================================================================
# 
# XGBoost predicted schedule: [3, 5, 8, 8, 8] for gen_length=32
# steps_per_block=1 (HARD-CODED: decode all tokens in block at once)
# 
# Call 1: generate_block_incremental(x, block_start=0, block_size=3)
#   Input:  x = [prompt, M, M, M, M, M, M, M, M, M, M, M, M, ...]  (M = masked)
#   Process: 
#     - One forward pass through model
#     - Unmask 3 highest-confidence tokens in positions [0, 3) in ONE SHOT
#     - Since XGBoost said "generate 3 tokens", we unmask exactly 3
#   Output: x = [prompt, 5, 2, 7, M, M, M, M, M, M, M, M, M, ...]
#           ^ Tokens at positions 0-2 unmasked
# 
# Call 2: generate_block_incremental(x, block_start=3, block_size=5)
#   Input:  x = [prompt, 5, 2, 7, M, M, M, M, M, M, M, M, M, ...]
#           ^ Positions 0-2 ALREADY unmasked from Call 1!
#   Process: 
#     - One forward pass through model
#     - Unmask 5 highest-confidence tokens in positions [3, 8) in ONE SHOT
#     - mask_index[0:3] = False (already unmasked) → positions 0-2 preserved!
#     - x0 = torch.where(mask_index, x0, x)  # Keeps positions 0-2 unchanged!
#   Output: x = [prompt, 5, 2, 7, 1, 9, 4, 3, 2, M, M, M, M, ...]
#           ^ Positions 0-7 now unmasked, positions 0-2 NOT regenerated
# 
# Call 3: generate_block_incremental(x, block_start=8, block_size=8)
#   Input:  x = [prompt, 5, 2, 7, 1, 9, 4, 3, 2, M, M, M, M, ...]
#           ^ Positions 0-7 ALREADY unmasked from Calls 1-2!
#   Process:
#     - One forward pass through model
#     - Unmask 8 highest-confidence tokens in positions [8, 16) in ONE SHOT
#     - mask_index[0:8] = False (already unmasked) → positions 0-7 preserved!
#   Output: x = [prompt, 5, 2, 7, 1, 9, 4, 3, 2, 8, 1, 4, 2, 9, 7, 3, M, ...]
#           ^ Positions 0-15 now unmasked, positions 0-7 NOT regenerated
# 
# ... and so on for remaining blocks
# 
# KEY INSIGHTS: 
# - XGBoost determines block_size → we unmask exactly that many tokens
# - steps_per_block=1 → decode all tokens in block with ONE forward pass (no refinement)
# - Much more efficient than generate_vanilla's iterative confidence-based refinement
# ============================================================================


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # This print is in main section, keep it for debugging
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
