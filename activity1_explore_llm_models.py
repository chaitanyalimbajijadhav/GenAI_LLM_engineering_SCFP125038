import json

MODELS_DATA = [
    {
        "category": "Small Model (< 1B parameters)",
        "name": "Qwen3-0.6B",
        "repo_id": "Qwen/Qwen3-0.6B",
        "creator": "Qwen Team (Alibaba Cloud)",
        "total_parameters": "0.6B (0.44B non-embedding parameters, 28 Transformer layers)",
        "is_generative": True,
        "architecture_and_capabilities": (
            "Causal Language Model (Dense). Supports reasoning, instruction following, "
            "dialogue, creative writing, coding, and multilingual tasks."
        ),
        "context_length": "32,768 tokens",
        "local_execution_suitable": True,
        "local_execution_notes": (
            "Highly suitable for local execution on consumer laptops/CPUs or entry-level GPUs "
            "due to its 0.6B parameter footprint."
        ),
        "recommended_hardware": (
            "No specific GPU specified on the model card; refers users to Qwen documentation "
            "and GitHub for inference benchmarks."
        ),
    },
    {
        "category": "Medium Model (1B–10B parameters)",
        "name": "Llama-3.1-8B",
        "repo_id": "meta-llama/Llama-3.1-8B",
        "creator": "Meta",
        "total_parameters": "8B",
        "is_generative": True,
        "architecture_and_capabilities": (
            "Autoregressive optimized Transformer language model. Supports multilingual text "
            "generation, coding, and assistant dialogue."
        ),
        "context_length": "128,000 tokens (128K)",
        "local_execution_suitable": True,
        "local_execution_notes": (
            "Suitable for local execution on capable consumer GPUs (e.g., 8GB–16GB+ VRAM) "
            "especially when using quantized weights (GGUF/AWQ/GPTQ)."
        ),
        "recommended_hardware": (
            "No consumer GPU specified on model card; reports training on NVIDIA H100-80GB "
            "GPUs (~1.46M GPU hours, 700W TDP)."
        ),
    },
    {
        "category": "Large Model (> 10B parameters)",
        "name": "Kolibri-1",
        "repo_id": "Aleph-Alpha/Kolibri-1",
        "creator": "Aleph Alpha Research GmbH / Aleph Alpha GmbH",
        "total_parameters": "78B total (3.46B active parameters per token via Mixture-of-Experts)",
        "is_generative": True,
        "architecture_and_capabilities": (
            "Mixture-of-Experts (MoE) generative reasoning model focused on German and English. "
            "Supports multi-step reasoning, coding, RAG, structured extraction, and tool calling."
        ),
        "context_length": "1,048,576 tokens (1M max; 262,144 tokens recommended for serving efficiency)",
        "local_execution_suitable": False,
        "local_execution_notes": (
            "Not suitable for standard laptops/consumer PCs; requires ~78 GB VRAM even in FP8 precision."
        ),
        "recommended_hardware": (
            "Minimum: 2x NVIDIA A100 80GB, 2x H100 SXM5, 1x H200, 1x B200, or 1x B300. "
            "Recommended: 2x H100 SXM5, 2x H200, 1x B200, or 1x B300."
        ),
    },
]


def classify_size(param_billions: float) -> str:
    if param_billions < 1.0:
        return "Small (< 1B)"
    elif param_billions <= 10.0:
        return "Medium (1B–10B)"
    return "Large (> 10B)"


def display_report(models):
    print("=" * 80)
    print("ACTIVITY 1 — EXPLORE LLM MODELS ON HUGGING FACE")
    print("=" * 80)
    print("Objective: Compare small, medium, and large Hugging Face language models across")
    print("creators, parameter counts, generative capabilities, context length, and hardware.
")

    for idx, m in enumerate(models, start=1):
        print(f"{idx}. {m['category']} — {m['name']} ({m['repo_id']})")
        print("-" * 80)
        print(f"  1. Creator               : {m['creator']}")
        print(f"  2. Parameters            : {m['total_parameters']}")
        print(f"  3. Generative Model      : {'Yes' if m['is_generative'] else 'No'} — {m['architecture_and_capabilities']}")
        print(f"  4. Context Length        : {m['context_length']}")
        print(f"  5. Local Execution       : {'Yes' if m['local_execution_suitable'] else 'Limited (Enterprise GPU required)'} — {m['local_execution_notes']}")
        print(f"  6. Recommended Hardware  : {m['recommended_hardware']}")
        print()


def print_comparison_table(models):
    print("=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Model':<26} | {'Creator':<18} | {'Params':<16} | {'Context':<15} | {'Local Fit'}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Qwen/Qwen3-0.6B", "Qwen Team", "0.6B", "32,768", "Yes (Lightweight)"),
        ("meta-llama/Llama-3.1-8B", "Meta", "8B", "128,000", "Yes (Quantized/GPU)"),
        ("Aleph-Alpha/Kolibri-1", "Aleph Alpha", "78B (3.46B act)", "1,048,576 (1M)", "No (~78GB FP8 VRAM)"),
    ]
    for r in rows:
        print(f"{r[0]:<26} | {r[1]:<18} | {r[2]:<16} | {r[3]:<15} | {r[4]}")
    print("=" * 80)


def fetch_live_hf_metadata(repo_ids):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print("
[Optional Live Check via huggingface_hub]")
        for repo in repo_ids:
            info = api.model_info(repo)
            print(f"  - {repo}: pipeline_tag={info.pipeline_tag}, downloads={info.downloads}, likes={info.likes}")
    except Exception as e:
        print(f"
[Live Hugging Face Hub check skipped]: {e}")


if __name__ == "__main__":
    display_report(MODELS_DATA)
    print_comparison_table(MODELS_DATA)
    fetch_live_hf_metadata([m["repo_id"] for m in MODELS_DATA])
