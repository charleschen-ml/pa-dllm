# Inference

import shutil
import evaluate
import json
import csv
import argparse
from math import ceil
from tqdm import tqdm
import torch
from peft import get_peft_model, PeftModel
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
# Ensure inference quantization config matches that of QAT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # To fix torch deterministic error
torch.use_deterministic_algorithms(True)

# Custom arguments for inference-specific parameters
class InferenceArguments:
    def __init__(self, 
                 eval_json_path="/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json",
                 adapter_path="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat",
                 output_csv_path="/content/drive/MyDrive/Colab_Notebooks/eic_llm/inference_output.csv",
                 bitwise_lora_adapter_path="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt",
                 use_quantization=True,
                 use_bitwise_lora=True,
                 bit_choices="32",
                 max_inf_size=100,
                 quant_layers="6,11",
                 inf_bit_config={},
                 default_bit=32):
        self.eval_json_path = eval_json_path
        self.adapter_path = adapter_path
        self.output_csv_path = output_csv_path
        self.bitwise_lora_adapter_path = bitwise_lora_adapter_path
        self.use_quantization = use_quantization
        self.use_bitwise_lora = use_bitwise_lora
        self.bit_choices = bit_choices
        self.max_inf_size = max_inf_size
        self.quant_layers = quant_layers
        self.inf_bit_config = inf_bit_config
        self.default_bit = default_bit

# Score squad metrics (EM, F1) after inference
def score_squad(predictions, references):
    for i in range(min(len(predictions), 2)): # print at most 2 samples
        print(f"prediction {i} = {predictions[i]['prediction_text']}")
        print(f"reference {i} = {references[i]['answers']['text']}")
    metric = evaluate.load("squad")
    results = metric.compute(predictions=predictions, references=references)

    num_correct = int(results["exact_match"] * len(predictions) / 100)
    print(f"Exact Match: {results['exact_match']:.2f} ({num_correct}/{len(predictions)})")
    print(f"F1 Score: {results['f1']:.2f}")
    return results

# Save inference outputs to csv
def save_predictions_to_csv(predictions, references, output_csv_path):
    metric = evaluate.load("squad")
    rows = []

    for pred, ref in zip(predictions, references):
        pred_text = pred["prediction_text"]
        ref_texts = ref["answers"]["text"]
        joined_refs = ", ".join([r.strip() for r in ref_texts])

        score = metric.compute(
            predictions=[{"prediction_text": pred_text, "id": pred["id"]}],
            references=[{"answers": {"text": ref_texts, "answer_start": [0] * len(ref_texts)}, "id": ref["id"]}]
        )

        rows.append({
            "prediction": pred_text,
            "reference": joined_refs,
            "exact_match": score["exact_match"],
            "f1_score": score["f1"]
        })
        rows.sort(key=lambda x: x["f1_score"])  # sort by worst predictions

    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prediction", "reference", "exact_match", "f1_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} results to {output_csv_path}")

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("inference", help="Run the inference script", dataclass_types=dataclass_types)
    else:
        parser = HfArgumentParser(dataclass_types)
    
    # Add inference-specific arguments
    parser.add_argument("--eval_json_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json",
                       help="Path to evaluation JSON file")
    parser.add_argument("--adapter_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat",
                       help="Path to adapter directory")
    parser.add_argument("--output_csv_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/eic_llm/inference_output.csv",
                       help="Path to save inference output CSV")
    parser.add_argument("--bitwise_lora_adapter_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt",
                       help="Path to bitwise LoRA adapter file")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Whether to apply quantization")
    parser.add_argument("--no_quantization", dest="use_quantization", action="store_false",
                       help="Disable quantization")
    parser.add_argument("--use_bitwise_lora", action="store_true", default=True,
                       help="Whether to use bitwise LoRA adapters")
    parser.add_argument("--no_bitwise_lora", dest="use_bitwise_lora", action="store_false",
                       help="Disable bitwise LoRA adapters")
    parser.add_argument("--bit_choices", type=str, default="32",
                       help="Comma-separated list of bit choices for LoRA")
    parser.add_argument("--max_inf_size", type=int, default=100,
                       help="Maximum number of examples to infer")
    parser.add_argument("--quant_layers", type=str, default="6,11",
                       help="Comma-separated list of h.* layers to quantize")
    parser.add_argument("--inf_bit_config", type=str, default=None,
                       help="JSON string for inference bit configuration (e.g., '{\"transformer.h.0\": 8, \"transformer.h.1\": 4}'). Default: 32 bits for all layers")
    parser.add_argument("--default_bit", type=int, default=32,
                       help="Default bit for all layers")
    return parser

def main(script_args, model_args, inference_args):
    """
    Args:
        script_args, training_args, model_args: Standard TRL arguments
        inference_args: InferenceArguments object containing inference-specific parameters
    """
    # Convert string arguments to lists if they're strings
    if isinstance(inference_args.bit_choices, str):
        bit_choices = [int(x.strip()) for x in inference_args.bit_choices.split(",")]
    else:
        bit_choices = inference_args.bit_choices
        
    if isinstance(inference_args.quant_layers, str):
        quant_layers = [int(x.strip()) for x in inference_args.quant_layers.split(",")]
    else:
        quant_layers = inference_args.quant_layers
    
    # Load validation examples from JSON
    # with open(inference_args.eval_json_path, "r") as f:
    #     dataset = [json.loads(line) for line in f][:inference_args.max_inf_size]
    # print(f"Examples used for inference: {len(dataset)}")

    ################
    # Model & Tokenizer
    ################
    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )
    quantization_config = get_quantization_config(model_args)
    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype=torch_dtype,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side="left", 
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # load base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
    ).to(device)
    print(f"Loaded base model path: {model_args.model_name_or_path}")

    ################
    # Inference
    ################

    # Inference loop
    predictions, references = [], []

    print("\nINFERENCE:\n")

    question = "What is 2*2?"
    prompt = f"{question}\n"
    print(f"prompt = \n{prompt}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=False, # required to disable KV cache
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    # generated_truncated = generated.split("\n")[0].strip()

    print("generated=\n{generated}")

    predictions.append({
        "id": qid,
        "prediction_text": generated
    })

    references.append({
        "id": qid,
        "answers": example["answers"]
    }) 
    return 0

if __name__ == "__main__":
    # Parse HF script_args, model_args
    parser = make_parser()
    script_args, model_args = parser.parse_args_into_dataclasses()
    
    # Parse custom inference args
    args = parser.parse_args()
    bit_choices = [int(x.strip()) for x in args.bit_choices.split(",")] # convert to list
    quant_layers = [int(x.strip()) for x in args.quant_layers.split(",")] # convert to list
    inference_args = InferenceArguments(
        eval_json_path=args.eval_json_path,
        adapter_path=args.adapter_path,
        output_csv_path=args.output_csv_path,
        bitwise_lora_adapter_path=args.bitwise_lora_adapter_path,
        use_quantization=args.use_quantization,
        use_bitwise_lora=args.use_bitwise_lora,
        bit_choices=args.bit_choices,
        max_inf_size=args.max_inf_size,
        quant_layers=args.quant_layers,
        inf_bit_config=args.inf_bit_config,
        default_bit=args.default_bit
    )
    
    # Run inference
    main(script_args, model_args, inference_args)
