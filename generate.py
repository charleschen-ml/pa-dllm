import torch
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

def extract_boxed(text):
    match = re.search(r'\\boxed{(\d+)}', text)
    return int(match.group(1)) if match else None

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

# @ torch.no_grad()
# def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''

#     # Create x = prompt + completion 
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone() # initialize prompt, while leaving completion tokens as <mask>

#     prompt_index = (x != mask_id) # create boolean mask with prompt = T, completion = F
#                                 # e.g. [T, T, T, ..., F, F, F, ...]
#                                 # used later if cfg enabled

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks # convert total_steps to steps_per_block

#     first_correct_step = None  # Track first step with correct answer
#     for num_block in range(num_blocks):

#         # initialize boolean mask to all <mask> in current block
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)

#         # calculate number of tokens to unmask at each step
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

#         for i in range(steps):
#             total_step = num_block * steps + i + 1 # total steps as efficiency metric
            
#             mask_index = (x == mask_id) # update the boolean mask (since last step)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits # get logits with current x

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # get index of token with highest logit at each position

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1) # convert logits to probs
                
#                 # extract prob at each position with highest logit
#                 x0_p = torch.squeeze( 
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             # mask out tokens beyond the current block
#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#             # torch.where(mask, tensor_A, tensor_B): if mask_index is True, use tensor A, otherwise use tensor B
#             # if token is true (masked), use x0 (token index with highest logit)
#             # otherwise use x (original token)
#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]): # loop through each batch
#                 # torch.topk(input, k): selects the top k tokens from "input" (list)
#                 # returns (values, indices)
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                
#                 # use "advanced indexing" to set all indices in select_index
#                 # equivalent to saying:
#                 # for index in select_index:
#                 #   transfer_index[j, index] = True
#                 transfer_index[j, select_index] = True

#             # unmask (freeze) the tokens in x (also using advanced indexing)
#             x[transfer_index] = x0[transfer_index]
            
#             # Store confidence for this block if this is the last step of the block
#             if i == steps_per_block - 1:  # Last step of this block

#                 block_confidence = []
#                 for j in range(block_size):
#                     token_pos = prompt.shape[1] + block_start + j
#                     if token_pos < confidence.shape[1]:
#                         conf_val = confidence[0, token_pos].item()

#                         if conf_val != -np.inf:  # Only include non-masked tokens
#                             block_confidence.append(conf_val)

#                 if block_confidence:
#                     block_confidences[num_block] = block_confidence


#             # check answer correct
#             out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
#             print("\n" + out_text)
#             is_correct = extract_boxed(out_text) == 72
#             if is_correct and first_correct_step is None:
#                 first_correct_step = total_step
#             print(f"{'✅' if is_correct else '❌'} | step: {total_step}")

#     print(f"\nFirst correct answer found at step: {first_correct_step if first_correct_step is not None else 'Never'}")
#     return x

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
        expected_answer: Optional, if set will compare against `extract_boxed()` for correctness logging.
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

@ torch.no_grad()
def generate_custom(model, tokenizer, prompt, steps=128, gen_length=128, block_sizes=None, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, curr_pos=0, correct_answer=None):
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

    first_correct_step = None  # Track first step with correct answer
    block_confidences = {}  # Track confidence for each block
    # Track per-step confidences (decoded vs remaining) to print at the end
    per_step_logs = []
    
    # Initialize confidence and entropy - will be captured at curr_pos block
    initial_entropy = None
    initial_confidence = None
    
    if cfg_scale > 0.:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        logits = model(x_).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x).logits
    
    def _entropy(v: float) -> float:
        return round(-float(v) * float(np.log(max(v, 1e-12))), 4)

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
            print(f"📝 Current generation state: '{current_decoded}'")
            
            # Count how many tokens are still masked
            gen_start = prompt.shape[1]
            gen_end = gen_start + gen_length
            masked_count = (x[0, gen_start:gen_end] == mask_id).sum().item()
            decoded_count = gen_length - masked_count
            print(f"🎭 Tokens decoded so far: {decoded_count}/{gen_length} (masked: {masked_count})")
            
            p = F.softmax(logits, dim=-1)
            initial_conf = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(torch.argmax(logits, dim=-1), -1)), -1
            )[0, gen_start:gen_end]
            
            initial_confidence = [round(float(initial_conf[i]), 4) for i in range(gen_length)]
            initial_entropy = [_entropy(float(initial_conf[i])) for i in range(gen_length)]
            
            # Show sample of captured values
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
            def _entropy(v: float) -> float:
                return round(-float(v) * float(np.log(max(v, 1e-12))), 4)

            decoded_entropy = [_entropy(float(all_conf[idx])) for idx in decoded_indices]
            remaining_entropy = [_entropy(float(all_conf[idx])) for idx in remaining_mask_indices]

            per_step_logs.append({
                'step': int(total_step),
                'block': int(num_block),
                'decoded_pos': decoded_indices.detach().cpu().tolist(),
                'remaining_pos': remaining_mask_indices.detach().cpu().tolist(),
                'decoded_conf': decoded_conf,
                'remaining_conf': remaining_conf,
                'decoded_entropy': decoded_entropy,
                'remaining_entropy': remaining_entropy,
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
                extracted_answer = extract_boxed(out_text)
                is_correct = extracted_answer == correct_answer
                if is_correct and first_correct_step is None:
                    first_correct_step = total_step
            # print(f"{'✅' if is_correct else '❌'} | step: {total_step}")

    print(f"\nFirst correct answer found at step: {first_correct_step if first_correct_step is not None else float('inf')}")

    # Print per-step confidence breakdown at the end
    if per_step_logs:
        print(f"\n{'='*60}")
        print("PER-STEP CONFIDENCE BREAKDOWN (decoded | remaining)")
        print(f"{'='*60}")
        for log in per_step_logs:
            print(f"step {log['step']} (block {log['block']}):")
            print(f"  top confidence: {log['decoded_conf']} {log['remaining_conf']}")
            print(f"  entropy: {log['decoded_entropy']} {log['remaining_entropy']}")
        print(f"{'='*60}")

    # block_confidences: Final confidence scores for tokens that were actually decoded in each block
    return x, first_correct_step if first_correct_step is not None else float('inf'), block_confidences, initial_entropy, initial_confidence

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
