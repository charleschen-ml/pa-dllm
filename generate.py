import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from inference import extract_boxed

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


@ torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    # Create x = prompt + completion 
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone() # initialize prompt, while leaving completion tokens as <mask>

    prompt_index = (x != mask_id) # create boolean mask with prompt = T, completion = F
                                # e.g. [T, T, T, ..., F, F, F, ...]
                                # used later if cfg enabled

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks # convert total_steps to steps_per_block

    first_correct_step = None  # Track first step with correct answer
    for num_block in range(num_blocks):

        # initialize boolean mask to all <mask> in current block
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)

        # calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            total_step = num_block * steps + i + 1 # total steps as efficiency metric
            
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

            # mask out tokens beyond the current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

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

            # unmask (freeze) the tokens in x (also using advanced indexing)
            x[transfer_index] = x0[transfer_index]

            # check answer correct
            out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            print("\n" + out_text)
            is_correct = extract_boxed(out_text) == 72
            if is_correct and first_correct_step is None:
                first_correct_step = total_step
            print(f"{'✅' if is_correct else '❌'} | step: {total_step}")

    print(f"\nFirst correct answer found at step: {first_correct_step if first_correct_step is not None else 'Never'}")
    return x


@ torch.no_grad()
def generate_custom(model, tokenizer, prompt, steps=128, gen_length=128, block_sizes=None, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336):
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
    '''
    # debug
    print("\nstart generate_custom:\n")

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

    first_correct_step = None  # Track first step with correct answer
    # Calculate cumulative block positions
    block_starts = [0] + [sum(block_sizes[:i]) for i in range(1, len(block_sizes))]
    
    for num_block in range(num_blocks):
        block_size = block_sizes[num_block]
        block_start = block_starts[num_block]
        block_end = block_start + block_size

        print(f"Starting block {num_block+1}/{num_blocks} (size: {block_size})")

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

            # mask out tokens beyond the current block
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

            # unmask (freeze) the tokens in x (also using advanced indexing)
            x[transfer_index] = x0[transfer_index]

            # check answer correct
            out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            print("\n" + out_text)
            is_correct = extract_boxed(out_text) == 72
            if is_correct and first_correct_step is None:
                first_correct_step = total_step
            print(f"{'✅' if is_correct else '❌'} | step: {total_step}")

    print(f"\nFirst correct answer found at step: {first_correct_step if first_correct_step is not None else 'Never'}")
    return x


def calculate_block_sizes(gen_length, base_block_length, sweep_position, sweep_value):
    '''
    Calculate block sizes for sweeping experiments.
    
    Args:
        gen_length: Total generation length
        base_block_length: Base block length (e.g., 2)
        sweep_position: Which block to sweep (0-indexed)
        sweep_value: Value to set at sweep position
    
    Returns:
        List of block sizes that sum to gen_length
        
    Example:
        calculate_block_sizes(32, 2, 0, 8) -> [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        calculate_block_sizes(32, 2, 1, 8) -> [2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    '''
    # Calculate how many base blocks we need
    num_base_blocks = gen_length // base_block_length
    
    # Create list of base block sizes
    block_sizes = [base_block_length] * num_base_blocks
    
    # Validate sweep position
    if sweep_position >= len(block_sizes):
        raise ValueError(f"sweep_position ({sweep_position}) must be less than number of blocks ({len(block_sizes)})")
    
    # Calculate the adjustment needed
    adjustment = sweep_value - base_block_length
    
    # Check if adjustment is possible
    if adjustment > 0:
        # Need to reduce other blocks to accommodate larger sweep block
        remaining_blocks = len(block_sizes) - 1
        if adjustment > remaining_blocks * base_block_length:
            raise ValueError(f"sweep_value ({sweep_value}) too large for gen_length ({gen_length})")
        
        # Reduce blocks from the end: last block goes to 0 first, then second-to-last to 1, etc.
        blocks_to_reduce = adjustment
        j = len(block_sizes) - 1  # Start from last block
        while blocks_to_reduce > 0 and j >= 0:
            if j != sweep_position:
                # Reduce this block completely before moving to the next
                reduction = min(blocks_to_reduce, block_sizes[j])
                block_sizes[j] -= reduction
                blocks_to_reduce -= reduction
            j -= 1
        
        if blocks_to_reduce > 0:
            raise ValueError(f"Cannot accommodate sweep_value ({sweep_value}) with given parameters")
    
    # Set the sweep position
    block_sizes[sweep_position] = sweep_value
    
    # Validate total
    total = sum(block_sizes)
    if total != gen_length:
        raise ValueError(f"Calculated block sizes sum to {total}, expected {gen_length}")
    
    return block_sizes


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
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
