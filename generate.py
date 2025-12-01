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
        # num_match = re.search(r"(\d+(?:\.\d+)?)\s*$", text.strip())
        # num_str = num_match.group(1) if num_match else None
        num_str = None # charles temporarily disable for debugging

    if num_str is None:
        return None
    try:
        return int(num_str) if '.' not in num_str else float(num_str)
    except ValueError:
        return None

def predict_block_size_neural(
    model,
    tokenizer,
    x,
    curr_pos,
    gen_start,
    gen_length,
    block_size_offset=0,
    max_block_size=10,
    block_length=1,
    verbose=True
):
    """
    Predict block size using neural scheduler head (trained with model).
    
    Args:
        model: LLaDA model with scheduler_head
        tokenizer: Tokenizer for decoding (for debug prints)
        x: Current sequence tensor [batch_size, seq_len]
        curr_pos: Current position in generation (relative to gen_start)
        gen_start: Starting position of generation in full sequence (prompt length)
        gen_length: Total generation length
        block_size_offset: Conservative offset to subtract from predicted block_size
        max_block_size: Maximum allowed block size
        block_length: Fallback block size if scheduler head not available
        verbose: If True, print prediction details
    
    Returns:
        block_size: Final predicted block size (int)
        logits: Model logits output [batch_size, seq_len, vocab_size]
    """
    
    # Calculate absolute position in full sequence (prompt + generation)
    abs_pos = gen_start + curr_pos
    
    # Run forward pass with token_positions to get scheduler predictions
    token_positions = torch.tensor([abs_pos], device=x.device, dtype=torch.long)
    
    print(f"🔍 DEBUG generate.py: x.shape = {x.shape}")
    print(f"🔍 DEBUG generate.py: curr_pos = {curr_pos}, gen_start = {gen_start}, abs_pos = {abs_pos}")
    decoded_text = tokenizer.decode(x[0].cpu().tolist(), skip_special_tokens=False)
    print(f"🔍 DEBUG generate.py: decoded text = {decoded_text[:]}...")  # First 200 chars
    
    outputs = model(
        x,
        token_positions=token_positions,
        block_size_rel_labels=None  # No labels for inference
    )
    
    logits = outputs.logits
    
    # Extract scheduler prediction if available
    if hasattr(outputs, 'block_size_predictions') and outputs.block_size_predictions is not None:
        # Get prediction at absolute position
        predictions = outputs.block_size_predictions[0]  # [seq_len]
        block_size_rel = predictions[abs_pos].item()
        
        # Convert to absolute block size based on REMAINING tokens
        remaining_tokens = gen_length - curr_pos
        block_size_raw = block_size_rel * remaining_tokens
        block_size = int(round(block_size_raw))
        
        # Apply conservative offset (subtract to make block size smaller)
        block_size_before_offset = block_size
        block_size = block_size - block_size_offset
        
        # Ensure block_size is at least 1 and at most min(remaining_tokens, max_block_size)
        block_size_final = max(1, min(block_size, remaining_tokens, max_block_size))
        
        # Print prediction details
        if verbose:
            print(f"📊 PREDICTION (Neural Scheduler Head):")
            print(f"   block_size_rel (Neural output) = {block_size_rel:.4f}")
            print(f"   remaining_tokens               = {remaining_tokens}")
            print(f"   block_size_raw                 = {block_size_rel:.4f} × {remaining_tokens} = {block_size_raw:.2f}")
            print(f"   block_size (rounded)           = {block_size_before_offset}")
            if block_size_offset > 0:
                print(f"   block_size (after offset -{block_size_offset})  = {block_size}")
            print(f"   block_size_final (clamped 1-{max_block_size}) = {block_size_final}")
            print(f"{'='*80}\n")
        
        return block_size_final, logits
    
    else:
        # Fallback: scheduler head not available
        if verbose:
            print(f"⚠️  WARNING: Scheduler head not available, using fallback block_length={block_length}")
        
        remaining_tokens = gen_length - curr_pos
        block_size = min(block_length, remaining_tokens)
        block_size = max(1, block_size - block_size_offset)
        
        return block_size, logits


def predict_block_size_mlp_features(
    model,
    mlp_scheduler,
    tokenizer,
    x,
    curr_pos,
    gen_start,
    gen_length,
    scheduler,
    use_regression,
    block_size_offset=0,
    max_block_size=10,
    verbose=True,
    use_hidden_states=False,  # Set to True to use hidden states
    use_features=True,  # Set to False for hidden-only mode
    use_dual_stream=False  # Set to True for dual-stream model
):
    """
    Predict block size using external MLP (features-only, hidden-only, or dual-stream).
    
    Args:
        model: LLaDA model (for extracting hidden states if use_hidden_states=True)
        mlp_scheduler: External trained MLP model (FeatureOnlySchedulerMLP or DualStreamSchedulerMLP)
        tokenizer: Tokenizer for decoding (for debug prints and feature extraction)
        x: Current sequence tensor [batch_size, seq_len]
        curr_pos: Current position in generation (relative to gen_start)
        gen_start: Starting position of generation in full sequence (prompt length)
        gen_length: Total generation length
        scheduler: XGBoost model (used only for extracting features via predict_block_size_xgboost)
        use_regression: Compatibility parameter (not used, MLP uses classification)
        block_size_offset: Conservative offset to subtract from predicted block_size
        max_block_size: Maximum allowed block size
        verbose: If True, print prediction details
        use_hidden_states: If True, extract and use hidden states
        use_features: If True, extract and use XGBoost features
        use_dual_stream: If True, pass features and hidden_states separately (for DualStreamSchedulerMLP)
    
    Returns:
        block_size: Final predicted block size (int, one of 1/2/3/4)
        logits: Model logits output [batch_size, seq_len, vocab_size]
    """
    
    # Calculate absolute position in full sequence (prompt + generation)
    abs_pos = gen_start + curr_pos
    
    # Step 1: Run model forward pass to get logits (and optionally hidden states)
    outputs = model(
        x,
        output_hidden_states=use_hidden_states  # Only request if needed
    )
    
    logits = outputs.logits
    
    # Extract hidden states if requested
    if use_hidden_states:
        # Find the first mask token position (126336)
        mask_id = 126336
        first_mask_idx = next((i for i, tid in enumerate(x[0]) if tid == mask_id), abs_pos)
        
        # Get hidden states from last layer
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        if use_dual_stream:
            # For dual-stream: extract FULL hidden states (will be pooled by model)
            hidden_states_full = last_hidden_states[0, :, :].unsqueeze(0).float()  # [1, seq_len, hidden_size]
        else:
            # For single-stream: extract hidden state at generation position
            hidden_state_at_pos = last_hidden_states[0, first_mask_idx, :]  # [hidden_size]
    
    # Step 2: Extract XGBoost features using the existing feature extraction logic
    # extract_features is defined in this file (generate.py)
    remaining_tokens = gen_length - curr_pos
    
    (initial_confidence, initial_entropy, initial_shannon_entropy, 
     additional_features, ar_context_tokens) = extract_features(
        logits=logits,
        x=x,
        gen_start=gen_start,
        gen_length=gen_length,
        tokenizer=tokenizer,
        verbose=False,
        curr_pos=curr_pos
    )
    
    # Get features for current position
    features_dict = additional_features.get(curr_pos, {})
    
    # Add position_relative feature (not in extract_features output)
    position_relative = round(curr_pos / gen_length, 4)
    
    # Convert features dict to tensor (match the EXACT order used in training)
    # Must match FEATURE_COLS from train_scheduler_head_features.py
    feature_names = [
        'position_relative',  # Will be added manually
        # Confidence features (positions 0-9)
        'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9',
        # Shannon entropy features (positions 0-9)
        'shannon_entropy_0', 'shannon_entropy_1', 'shannon_entropy_2', 'shannon_entropy_3', 'shannon_entropy_4',
        'shannon_entropy_5', 'shannon_entropy_6', 'shannon_entropy_7', 'shannon_entropy_8', 'shannon_entropy_9',
        # Aggregate features
        'top1_margin', 'mean_confidence', 'shannon_mean_entropy',
        'conf_std', 'shannon_entropy_std',
        'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
    ]
    
    # Build feature vector
    feature_values = [position_relative]  # First feature
    for name in feature_names[1:]:  # Skip first since we added it manually
        feature_values.append(features_dict.get(name, 0.0))  # Use 0.0 as fallback
    
    xgb_features = torch.tensor(
        feature_values,
        dtype=torch.float32,
        device=mlp_scheduler.device if hasattr(mlp_scheduler, 'device') else torch.device('cuda')
    ).unsqueeze(0)  # [1, 30]
    
    # Step 3: Prepare inputs based on mode
    if use_dual_stream:
        # Dual-stream: pass features and hidden states separately
        # (will be processed by separate MLPs and fused)
        pass  # Will be handled in Step 4
    else:
        # Single-stream: combine features based on mode
        if use_hidden_states and use_features:
            # Both: concatenate features + hidden states
            combined_features = torch.cat([xgb_features, hidden_state_at_pos.unsqueeze(0)], dim=1)  # [1, 30+4096]
        elif use_hidden_states and not use_features:
            # Hidden-only: just hidden states
            combined_features = hidden_state_at_pos.unsqueeze(0).float()  # [1, 4096]
        elif not use_hidden_states and use_features:
            # Features-only: just XGBoost features
            combined_features = xgb_features  # [1, 30]
        else:
            raise ValueError("Must use at least one of: use_hidden_states or use_features")
    
    # DEBUG: Print feature/hidden state info
    if verbose:
        print(f"\n{'='*80}")
        if use_hidden_states and not use_features:
            print(f"🔍 DEBUG: Hidden States Only at curr_pos={curr_pos}")
        elif use_features and not use_hidden_states:
            print(f"🔍 DEBUG: Features Only at curr_pos={curr_pos}")
        else:
            print(f"🔍 DEBUG: Features + Hidden States at curr_pos={curr_pos}")
        print(f"{'='*80}")
        
        if use_features:
            # Print all 30 features in the same order as FEATURE_COLS from training
            feature_names_ordered = [
                'position_relative',
                'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9',
                'shannon_entropy_0', 'shannon_entropy_1', 'shannon_entropy_2', 'shannon_entropy_3', 'shannon_entropy_4',
                'shannon_entropy_5', 'shannon_entropy_6', 'shannon_entropy_7', 'shannon_entropy_8', 'shannon_entropy_9',
                'top1_margin', 'mean_confidence', 'shannon_mean_entropy',
                'conf_std', 'shannon_entropy_std',
                'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
            ]
            
            for i, feat_name in enumerate(feature_names_ordered, 1):
                if feat_name == 'position_relative':
                    val = position_relative
                else:
                    val = features_dict.get(feat_name, -999.0)
                print(f"   [{i:2d}] {feat_name:<25} {val:.4f}")
        
        if use_hidden_states:
            if use_dual_stream:
                print(f"\n   Hidden state stats (full sequence):")
                print(f"      Shape: {hidden_states_full.shape}")
                print(f"      Mean: {hidden_states_full.mean().item():.4f}")
                print(f"      Std:  {hidden_states_full.std().item():.4f}")
            else:
                print(f"\n   Hidden state stats (single position):")
                print(f"      Mean: {hidden_state_at_pos.mean().item():.4f}")
                print(f"      Std:  {hidden_state_at_pos.std().item():.4f}")
                print(f"      Min:  {hidden_state_at_pos.min().item():.4f}")
                print(f"      Max:  {hidden_state_at_pos.max().item():.4f}")
            print(f"\n   Position info:")
            print(f"      gen_start: {gen_start}")
            print(f"      curr_pos: {curr_pos}")
            print(f"      first_mask_idx: {first_mask_idx}")
        
        print(f"{'='*80}\n")
    
    # Step 4: Run MLP to get class predictions
    with torch.no_grad():
        if use_dual_stream:
            # Dual-stream: pass features and hidden states separately
            class_logits = mlp_scheduler(xgb_features, hidden_states_full)  # [1, 4]
        else:
            # Single-stream: pass combined features
            class_logits = mlp_scheduler(combined_features)  # [1, 4]
        
        predicted_class = torch.argmax(class_logits, dim=1).item()  # 0, 1, 2, or 3
        predicted_block_size = predicted_class + 1  # Convert to 1, 2, 3, or 4
    
    # Apply conservative offset (subtract to make block size smaller)
    block_size_before_offset = predicted_block_size
    predicted_block_size = predicted_block_size - block_size_offset
    
    # Ensure block_size is at least 1 and at most min(remaining_tokens, max_block_size)
    block_size_final = max(1, min(predicted_block_size, remaining_tokens, max_block_size))
    
    # Print prediction details
    if verbose:
        class_probs = torch.softmax(class_logits, dim=1)[0]  # [4]
        
        # Determine mode description
        if use_dual_stream:
            mode_desc = "Dual-Stream MLP (Features + Hidden States with Fusion)"
        elif use_hidden_states and use_features:
            mode_desc = "MLP (Features + Hidden States)"
        elif use_hidden_states:
            mode_desc = "MLP (Hidden States Only)"
        else:
            mode_desc = "MLP (Features Only)"
        
        print(f"📊 PREDICTION ({mode_desc}):")
        print(f"   Class probabilities: [1tok: {class_probs[0]:.3f}, 2tok: {class_probs[1]:.3f}, "
              f"3tok: {class_probs[2]:.3f}, 4tok: {class_probs[3]:.3f}]")
        print(f"   Predicted class: {predicted_class} → {predicted_block_size} tokens")
        print(f"   remaining_tokens = {remaining_tokens}")
        if block_size_offset > 0:
            print(f"   block_size (after offset -{block_size_offset}) = {predicted_block_size}")
        print(f"   block_size_final (clamped 1-{max_block_size}) = {block_size_final}")
        print(f"{'='*80}\n")
    
    return block_size_final, logits


def predict_block_size_xgboost(
    logits,
    x,
    gen_start,
    gen_length,
    tokenizer,
    curr_pos,
    scheduler,
    use_regression=True,
    block_size_offset=0,
    max_block_size=10,
    block_length=1,
    verbose=True,
    baseline_strategy=None
):
    """
    Predict block size using XGBoost scheduler with extracted features.
    
    Args:
        logits: Model logits output [batch_size, seq_len, vocab_size]
        x: Current sequence tensor [batch_size, seq_len]
        gen_start: Starting position of generation
        gen_length: Total generation length
        tokenizer: Tokenizer for feature extraction
        curr_pos: Current position in generation
        scheduler: XGBoost model (or None for fallback)
        use_regression: If True, use XGBRegressor; else XGBClassifier
        block_size_offset: Conservative offset to subtract from predicted block_size
        max_block_size: Maximum allowed block size
        block_length: Fallback block size if scheduler is None
        verbose: If True, print prediction details
    
    Returns:
        block_size: Final predicted block size (int)
    """
    
    if scheduler is not None or baseline_strategy is not None:
        # Extract features from logits (or use baseline strategy)
        from inference import predict_block_size
        
        # Hash the prompt tokens to create question-specific seed for random baselines
        # This makes random patterns vary by question but stay deterministic across runs
        prompt_tokens = x[0, :gen_start].cpu().tolist()  # Extract prompt tokens
        question_hash = hash(tuple(prompt_tokens)) & 0x7FFFFFFF  # Positive int hash
        
        # Skip expensive feature extraction if using baseline strategy
        if baseline_strategy is None:
            (initial_confidence, initial_entropy, initial_shannon_entropy, 
             additional_features, ar_context_tokens) = extract_features(
                logits=logits,
                x=x,
                gen_start=gen_start,
                gen_length=gen_length,
                tokenizer=tokenizer,
                verbose=False,
                curr_pos=curr_pos
            )
            
            position_relative = round(curr_pos / gen_length, 4)
            features = additional_features.get(curr_pos, {})
            features['position_relative'] = position_relative
        else:
            # Baseline mode: no features needed
            features = {}
        
        # Calculate remaining tokens for accurate prediction
        remaining_tokens = gen_length - curr_pos
        
        # Get relative block size from XGBoost
        block_size_rel = predict_block_size(
            scheduler=scheduler,
            features=features,
            gen_length=gen_length,
            use_regression=use_regression,
            remaining_length=remaining_tokens,
            max_block_size=max_block_size,
            baseline_strategy=baseline_strategy,
            question_hash=question_hash
        )
        block_size_raw = block_size_rel * remaining_tokens
        block_size = int(round(block_size_raw))
        
        # Apply conservative offset (subtract to make block size smaller)
        block_size_before_offset = block_size
        block_size = block_size - block_size_offset
        
        # Ensure block_size is at least 1 and at most min(remaining_tokens, max_block_size)
        # Cap at max_block_size to match training data distribution
        block_size_final = max(1, min(block_size, remaining_tokens, max_block_size))
        
        # Print prediction details
        if verbose:
            print(f"📊 PREDICTION:")
            if baseline_strategy:
                print(f"   baseline_strategy               = {baseline_strategy}")
            print(f"   block_size_rel (predicted)      = {block_size_rel:.4f}")
            print(f"   remaining_tokens                = {remaining_tokens}")
            print(f"   block_size_raw                  = {block_size_rel:.4f} × {remaining_tokens} = {block_size_raw:.2f}")
            print(f"   block_size (rounded)            = {block_size_before_offset}")
            if block_size_offset > 0:
                print(f"   block_size (after offset -{block_size_offset}) = {block_size}")
            print(f"   block_size_final (clamped 1-{max_block_size}) = {block_size_final}")
            print(f"{'='*80}\n")
        
        return block_size_final
    
    else:
        # Fallback: use fixed block_length
        remaining_tokens = gen_length - curr_pos
        block_size = min(block_length, remaining_tokens)
        
        # Apply conservative offset even for fallback mode
        block_size = max(1, block_size - block_size_offset)
        
        return block_size


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
    block_size_predictions = []  # Collect scheduler predictions

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
                outputs = model(x_)
                logits = outputs.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                # Note: scheduler predictions not extracted for CFG mode (would need special handling)
            else:
                outputs = model(x)
                logits = outputs.logits  # get logits with current x
                
                # Debug: Check what outputs contains (DISABLED)
                # print(f"🔍 DEBUG: outputs type = {type(outputs)}")
                # print(f"🔍 DEBUG: hasattr(outputs, 'block_size_predictions') = {hasattr(outputs, 'block_size_predictions')}")
                # if hasattr(outputs, 'block_size_predictions'):
                #     print(f"🔍 DEBUG: outputs.block_size_predictions = {outputs.block_size_predictions}")
                #     print(f"🔍 DEBUG: outputs.block_size_predictions is not None = {outputs.block_size_predictions is not None}")
                
                # Extract scheduler predictions if available
                if hasattr(outputs, 'block_size_predictions') and outputs.block_size_predictions is not None:
                    print(f"🔍 SCHEDULER PREDICTIONS AVAILABLE")
                    # Get predictions for all positions (shape: batch_size, seq_len)
                    preds = outputs.block_size_predictions[0]  # batch_size=1, so take first
                    
                    # Store predictions for masked positions
                    for pos_idx in range(len(preds)):
                        if mask_index[0, pos_idx]:  # Only for positions that are still masked
                            block_size_predictions.append({
                                'step': total_step,
                                'position': pos_idx,
                                'block_size_rel': preds[pos_idx].item()
                            })

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

    return x, block_size_predictions

@torch.no_grad()
def generate_charles(model, tokenizer, prompt, scheduler=None, steps=128, gen_length=128, block_length=128, 
                     temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, 
                     expected_answer=None, use_regression=True, block_size_offset=0, max_block_size=10,
                     scheduler_type='xgboost', baseline_strategy=None, mlp_scheduler=None):
    '''
    Dynamic block size generation using scheduler (XGBoost, Neural Scheduler Head, or MLP with Features).
    
    Args:
        model: Mask predictor.
        tokenizer: Tokenizer for decoding outputs.
        prompt: A tensor of shape (1, L).
        scheduler: Trained XGBoost model for predicting block sizes. If None, uses fixed block_length.
                   Not used if scheduler_type='neural' or 'mlp_features'.
        steps: Sampling steps (not used in dynamic version, kept for compatibility).
        gen_length: Generated answer length.
        block_length: Default block length if scheduler is None.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        expected_answer: Optional, if set will compare against `extract_numerical()` for correctness logging.
        use_regression: If True, scheduler is regression model; else classification (only for XGBoost).
        block_size_offset: Integer offset to subtract from predicted block_size for conservative sizing (default: 0).
        max_block_size: Maximum allowed block size to cap predictions (default: 10, matching training data cap).
        scheduler_type: Type of scheduler to use. 'xgboost', 'neural', or 'mlp_features' (default: 'xgboost').
                        'xgboost': Uses XGBoost model with extracted features
                        'mlp_features': Uses external MLP with XGBoost features + hidden states
                        'neural': Uses trained scheduler head in the model
        baseline_strategy: Baseline mode for testing. None (use scheduler), 'always_1', 'always_2', 'always_4', 'random_uniform' (default: None).
                           'random_uniform' splits equally across 1 to max_block_size (e.g., 25-25-25-25 for max=4).
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
    if scheduler_type == 'neural':
        print(f"   - Scheduler: Neural Scheduler Head (end-to-end learned)")
    elif scheduler_type == 'mlp_features':
        print(f"   - Scheduler: External MLP (XGBoost features + hidden states)")
    elif scheduler_type == 'xgboost':
        print(f"   - Scheduler: {'XGBoost' if scheduler is not None else 'Fixed block size'}")
    else:
        print(f"   - Scheduler: Unknown type '{scheduler_type}'")
    print(f"{'='*80}\n")

    prompt_index = (x != mask_id)  # create boolean mask with prompt = T, completion = F
                                   # e.g. [T, T, T, ..., F, F, F, ...]
                                   # used later if cfg enabled

    # Track current position and features for XGBoost
    curr_pos = 0
    predicted_block_sizes = []  # Track all predicted block sizes
    num_steps = 0  # Track total number of steps

    while curr_pos < gen_length:  # loop until we reach gen_length
        num_steps += 1
        # Get current generation region
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length

        # print the size of x and decoded x
        print(f"The size of x: {x.shape}")
        decoded_x = tokenizer.batch_decode(x, skip_special_tokens=False)[0]
        print(f"The decoded x:\n {decoded_x}")

        # Get logits and predict block_size (method depends on scheduler_type)
        if scheduler_type == 'neural':
            # Neural scheduler: get predictions from scheduler head
            if cfg_scale > 0.:
                # CFG not yet supported with neural scheduler
                raise NotImplementedError("CFG (cfg_scale > 0) not yet supported with neural scheduler")
            
            block_size, logits = predict_block_size_neural(
                model=model,
                tokenizer=tokenizer,
                x=x,
                curr_pos=curr_pos,
                gen_start=gen_start,
                gen_length=gen_length,
                block_size_offset=block_size_offset,
                max_block_size=max_block_size,
                block_length=block_length,
                verbose=True
            )
        
        elif scheduler_type == 'mlp_features':
            # External MLP scheduler: uses both XGBoost features AND hidden states
            if cfg_scale > 0.:
                # CFG not yet supported with MLP scheduler
                raise NotImplementedError("CFG (cfg_scale > 0) not yet supported with mlp_features scheduler")
            
            # Auto-detect mode from mlp_scheduler (set in run_inference.py)
            mlp_mode = getattr(mlp_scheduler, 'mlp_mode', 'hidden_only')
            use_dual_stream = getattr(mlp_scheduler, 'use_dual_stream', False)
            
            # Set flags based on mode
            if mlp_mode == 'dual_stream':
                use_hidden_states = True
                use_features = True
            elif mlp_mode == 'hidden_only':
                use_hidden_states = True
                use_features = False
            elif mlp_mode == 'features_only':
                use_hidden_states = False
                use_features = True
            else:
                # Fallback (shouldn't happen)
                use_hidden_states = True
                use_features = False
            
            block_size, logits = predict_block_size_mlp_features(
                model=model,
                mlp_scheduler=mlp_scheduler,
                tokenizer=tokenizer,
                x=x,
                curr_pos=curr_pos,
                gen_start=gen_start,
                gen_length=gen_length,
                scheduler=scheduler,
                use_regression=use_regression,
                block_size_offset=block_size_offset,
                max_block_size=max_block_size,
                verbose=True,
                use_hidden_states=use_hidden_states,
                use_features=use_features,
                use_dual_stream=use_dual_stream
            )
        
        elif scheduler_type == 'xgboost':
            # XGBoost scheduler: get logits first, then extract features
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits  # get logits with current x
            
            block_size = predict_block_size_xgboost(
                logits=logits,
                x=x,
                gen_start=gen_start,
                gen_length=gen_length,
                tokenizer=tokenizer,
                curr_pos=curr_pos,
                scheduler=scheduler,
                use_regression=use_regression,
                block_size_offset=block_size_offset,
                max_block_size=max_block_size,
                block_length=block_length,
                verbose=True,
                baseline_strategy=baseline_strategy
            )
        
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Must be 'neural', 'mlp_features', or 'xgboost'")

        # Generate tokens for current block (semi-AR: unmask next N tokens left-to-right)
        # Use the predicted tokens with added noise for sampling
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # # Hardcode block_size for first iteration only (for debugging)
        # if curr_pos == 0:
        #     block_size = 2
        # # After first iteration, use scheduler/default block_size

        # Print decoded x and x0
        # print(f"The decoded x:\n {tokenizer.batch_decode(x, skip_special_tokens=False)[0]}")
        # print(f"\nThe decoded x0 (FULL):\n {tokenizer.batch_decode(x0, skip_special_tokens=False)[0]}")
        
        # Show prompt region vs completion region separately
        x0_prompt = x0[:, :gen_start]
        x0_completion = x0[:, gen_start:gen_end]
        # print(f"\nx0 PROMPT region:\n {tokenizer.batch_decode(x0_prompt, skip_special_tokens=False)[0]}")
        
        # Show completion region with each token in quotes (with escaped special chars)
        print(f"\nx0 COMPLETION region (token by token):")
        completion_tokens = []
        for i in range(gen_length):
            token_id = x0_completion[0, i].item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            # Use repr() to show escaped characters like \n, \t, etc.
            completion_tokens.append(repr(token_text))
        print(" ".join(completion_tokens))
        
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
        predicted_block_sizes.append(block_size)  # Track this block size

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
        
        # Show each newly unmasked token in quotes (with escaped special chars)
        print(f"🆕 Newly unmasked tokens (token by token):")
        unmasked_tokens = []
        for i in range(block_size):
            token_id = newly_unmasked[0, i].item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            # Use repr() to show escaped characters like \n, \t, etc.
            unmasked_tokens.append(repr(token_text))
        print(" ".join(unmasked_tokens))
        
    # print final output
    # out_text = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    # print("\n" + out_text)

    # Print generation summary
    print(f"\n{'='*80}")
    print(f"✅ GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Total steps: {num_steps}")
    print(f"📊 Total tokens generated: {gen_length}")
    print(f"📊 Predicted block sizes: {predicted_block_sizes}")
    print(f"📊 Average block size: {sum(predicted_block_sizes)/len(predicted_block_sizes):.2f}")
    print(f"⚡ Speedup vs autoregressive: {gen_length/num_steps:.2f}x")
    print(f"{'='*80}\n")

    return x, num_steps


def _entropy(v: float) -> float:
    """Calculate self-information entropy for a single confidence value."""
    return round(-float(v) * float(np.log(max(v, 1e-12))), 4)


def _shannon_entropy(probs: torch.Tensor) -> float:
    """Calculate Shannon entropy over full probability distribution."""
    # Ensure probabilities are positive and normalized
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum()  # Normalize just in case
    entropy = -torch.sum(probs * torch.log(probs))
    return round(float(entropy), 4)


def extract_features(logits, x, gen_start, gen_length, tokenizer=None, verbose=False, curr_pos=None):
    """
    Extract confidence, entropy, and 30 additional features from logits.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        x: Current sequence tensor [batch_size, seq_len] (for AR context)
        gen_start: Starting position of generation region
        gen_length: Length of generation region
        tokenizer: Optional tokenizer for verbose printing
        verbose: If True, print feature extraction details
        curr_pos: Optional current position for verbose printing
        
    Returns:
        Tuple of (initial_confidence, initial_entropy, initial_shannon_entropy, 
                  additional_features, ar_context_tokens)
    """
    gen_end = gen_start + gen_length
    
    # CAPTURE TOP-1 TOKENS after AR context is built
    if verbose and curr_pos is not None:
        print(f"🔍 CAPTURING top-1 tokens after AR context (blocks 0 to {curr_pos-1} processed)")
    
    top1_tokens = torch.argmax(logits, dim=-1)  # Get top-1 predictions for all positions
    ar_context_tokens = x.clone()  # Start with current state (AR context + masks)
    ar_context_tokens[0, gen_start:gen_end] = top1_tokens[0, gen_start:gen_end]  # Fill with top-1 predictions
    
    if verbose and tokenizer is not None:
        ar_tokens_text = tokenizer.decode(ar_context_tokens[0, gen_start:gen_end], skip_special_tokens=True)
        print(f"📝 AR context + top-1 predictions: '{ar_tokens_text}'")
    
    # Calculate comprehensive confidence metrics
    p = F.softmax(logits, dim=-1)  # Full softmax probabilities
    gen_logits = logits[0, gen_start:gen_end]  # Logits for generation positions only
    gen_probs = p[0, gen_start:gen_end]  # Probabilities for generation positions only
    
    # Basic confidence and entropy
    initial_conf = torch.squeeze(
        torch.gather(gen_probs, dim=-1, index=torch.unsqueeze(torch.argmax(gen_logits, dim=-1), -1)), -1
    )
    
    initial_confidence = [round(float(initial_conf[i]), 4) for i in range(gen_length)]
    initial_entropy = [_entropy(float(initial_conf[i])) for i in range(gen_length)]
    initial_shannon_entropy = [_shannon_entropy(gen_probs[i]) for i in range(gen_length)]
    
    # Calculate additional features for each position (30 features total for XGBoost)
    additional_features = {}
    
    for pos in range(gen_length):
        # Global features (mean/std across remaining tokens from current position)
        remaining_conf = initial_confidence[pos:]
        remaining_shannon_entropy = initial_shannon_entropy[pos:]
        
        # Top1 margin (difference between top-1 and top-2) at this position
        pos_probs = gen_probs[pos]  # Probabilities for this position
        pos_top_probs, _ = torch.topk(pos_probs, k=min(2, pos_probs.shape[0]))
        if len(pos_top_probs) >= 2:
            top1_margin = float(pos_top_probs[0] - pos_top_probs[1])
        else:
            top1_margin = float(pos_top_probs[0])  # Only one token available
        
        # Aggregate statistics
        mean_confidence = float(np.mean(remaining_conf)) if remaining_conf else 0.0
        shannon_mean_entropy = float(np.mean(remaining_shannon_entropy)) if remaining_shannon_entropy else 0.0
        conf_std = float(np.std(remaining_conf)) if len(remaining_conf) > 1 else 0.0
        shannon_entropy_std = float(np.std(remaining_shannon_entropy)) if len(remaining_shannon_entropy) > 1 else 0.0
        
        # Top-K vs Sequential features
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
        
        # Build 30-feature dictionary matching predict_block_size() feature_order
        # conf_0 through conf_9: confidence values for positions pos to pos+9
        # shannon_entropy_0 through shannon_entropy_9: shannon entropy values for positions pos to pos+9
        feature_dict = {
            # Confidence features (positions pos to pos+9)
            'conf_0': remaining_conf[0] if len(remaining_conf) > 0 else 0.0,
            'conf_1': remaining_conf[1] if len(remaining_conf) > 1 else 0.0,
            'conf_2': remaining_conf[2] if len(remaining_conf) > 2 else 0.0,
            'conf_3': remaining_conf[3] if len(remaining_conf) > 3 else 0.0,
            'conf_4': remaining_conf[4] if len(remaining_conf) > 4 else 0.0,
            'conf_5': remaining_conf[5] if len(remaining_conf) > 5 else 0.0,
            'conf_6': remaining_conf[6] if len(remaining_conf) > 6 else 0.0,
            'conf_7': remaining_conf[7] if len(remaining_conf) > 7 else 0.0,
            'conf_8': remaining_conf[8] if len(remaining_conf) > 8 else 0.0,
            'conf_9': remaining_conf[9] if len(remaining_conf) > 9 else 0.0,
            # Shannon entropy features (positions pos to pos+9)
            'shannon_entropy_0': remaining_shannon_entropy[0] if len(remaining_shannon_entropy) > 0 else 0.0,
            'shannon_entropy_1': remaining_shannon_entropy[1] if len(remaining_shannon_entropy) > 1 else 0.0,
            'shannon_entropy_2': remaining_shannon_entropy[2] if len(remaining_shannon_entropy) > 2 else 0.0,
            'shannon_entropy_3': remaining_shannon_entropy[3] if len(remaining_shannon_entropy) > 3 else 0.0,
            'shannon_entropy_4': remaining_shannon_entropy[4] if len(remaining_shannon_entropy) > 4 else 0.0,
            'shannon_entropy_5': remaining_shannon_entropy[5] if len(remaining_shannon_entropy) > 5 else 0.0,
            'shannon_entropy_6': remaining_shannon_entropy[6] if len(remaining_shannon_entropy) > 6 else 0.0,
            'shannon_entropy_7': remaining_shannon_entropy[7] if len(remaining_shannon_entropy) > 7 else 0.0,
            'shannon_entropy_8': remaining_shannon_entropy[8] if len(remaining_shannon_entropy) > 8 else 0.0,
            'shannon_entropy_9': remaining_shannon_entropy[9] if len(remaining_shannon_entropy) > 9 else 0.0,
            # Aggregate features
            'top1_margin': top1_margin,
            'mean_confidence': mean_confidence,
            'shannon_mean_entropy': shannon_mean_entropy,
            'conf_std': conf_std,
            'shannon_entropy_std': shannon_entropy_std,
            'top4_conf_min': top4_conf_min,
            'next4_conf_min': next4_conf_min,
            'top8_conf_min': top8_conf_min,
            'next8_conf_min': next8_conf_min,
        }
        
        additional_features[pos] = feature_dict

    # Show sample of captured values
    if verbose:
        print(f"📊 Sample confidence values: {initial_confidence[:5]}...")
        print(f"📈 Sample entropy values: {initial_entropy[:5]}...")
        print(f"✅ Captured {len(initial_confidence)} confidence/entropy pairs")
        print("=" * 60)
    
    return initial_confidence, initial_entropy, initial_shannon_entropy, additional_features, ar_context_tokens


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

    first_correct_step = None  # Track first step with correct answer
    block_confidences = {}  # Track confidence for each block
    # Track per-step confidences (decoded vs remaining) to print at the end
    per_step_logs = []
    
    # Initialize confidence and entropy - will be captured at curr_pos block
    initial_entropy = None
    initial_confidence = None
    intermediate_x = None  # Capture x state at curr_pos for MLP training
    
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
            # Capture intermediate x state for MLP training
            intermediate_x = x.clone()
            
            if verbose:
                print(f"\n🎯 CAPTURING confidence/entropy at block {num_block} (curr_pos={curr_pos})")
                print(f"   Token position: {block_start} (block {num_block} starts here)")
                print(f"   Captured intermediate_x shape: {intermediate_x.shape}")
            
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
            
            # Extract features using the shared extract_features function
            # IMPORTANT: Pass block_start (token position) instead of curr_pos (block number)
            # This ensures features are indexed by token position consistently with generate_charles
            (initial_confidence, initial_entropy, initial_shannon_entropy, 
             additional_features, ar_context_tokens) = extract_features(
                logits=logits,
                x=x,
                gen_start=gen_start,
                gen_length=gen_length,
                tokenizer=tokenizer,
                verbose=verbose,
                curr_pos=block_start  # Use token position, not block number!
            )

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
    # if per_step_logs and verbose:
    #     print(f"\n{'='*60}")
    #     print("PER-STEP CONFIDENCE BREAKDOWN (decoded | remaining)")
    #     print(f"{'='*60}")
    #     for log in per_step_logs:
    #         print(f"step {log['step']} (block {log['block']}):")
    #         print(f"  top confidence: {log['decoded_conf']} {log['remaining_conf']}")
    #         print(f"  entropy: {log['decoded_entropy']} {log['remaining_entropy']}")
    #         if 'decoded_shannon_entropy' in log:
    #             print(f"  shannon entropy: {log['decoded_shannon_entropy']} {log['remaining_shannon_entropy']}")
    #     print(f"{'='*60}")

    # block_confidences: Final confidence scores for tokens that were actually decoded in each block
    return x, first_correct_step if first_correct_step is not None else float('inf'), block_confidences, initial_entropy, initial_confidence, ar_context_tokens, additional_features, initial_shannon_entropy, intermediate_x

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

    out = generate_vanilla(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # This print is in main section, keep it for debugging
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
