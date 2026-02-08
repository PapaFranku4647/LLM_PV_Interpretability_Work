# LLM Priors for ERM over Programs

Our work addresses the following question : 
> Can we design program-learning methods that are efficient in both samples and computation, avoiding exponential enumeration while requiring fewer samples than gradient-based training?

## Abstract

We study program-learning methods that are efficient in both samples and computation. Classical learning theory suggests that when the target admits a short program description (for example, a short piece of "Python code"), it can be learned from relatively few examples by performing ERM over the program class. However, this approach relies on enumerating candidate programs, which is typically exponential in the description length. In contrast, gradient-based training avoids explicit search, but for some families of short programs it can require exponentially many samples to succeed.

We propose **LLM-PV**, a propose-and-verify recipe that enables ERM-style selection over a discrete program class without exhaustive enumeration. A pretrained LLM induces a proposal distribution over candidate programs; each proposal is executed, scored on a held-out validation set, and the best program is selected. The method uses no gradient updates and does not use validation feedback to adapt the sampling distribution. Across algorithmic tasks including parity variants, pattern matching, and primality testing, LLM-PV often recovers the exact underlying rule from a small labeled set and generalizes far beyond the training sequence lengths. In the same regimes, SGD-trained transformers and standard adaptation baselines (fine-tuning and in-context learning), as well as classical ML baselines, can fit the training data yet fail to generalize reliably. Together, these results suggest that pretrained LLM priors can serve as effective search biases for ERM, narrowing the gap between statistical and computational efficiency.

## LLM-PV • In-Context • Fine-Tuning

The repository contains three types of experiments.

- **llm-pv** — Synthesize a Python function `f(x)` that matches the input-output relationship.
- **in_context_learning** — Run local models via **vLLM**; few-shot prompts with hundreds of examples; batched inference and accuracy summaries.
- **finetuning** — Train from scratch or finetune LMs (e.g., Qwen/Llama/DeepSeek) as **binary classifiers** on the same tasks; YAML sweeps included.

## Quickstart

```bash
# 1) Create environment (Conda)
conda env create -f environment.yaml
conda activate llm_pv

# 2) (Not Required) Already included in environment.yaml
pip install -r requirements.txt
```

### LLM-PV (OpenAI)
```bash
export OPENAI_API_KEY=sk-...
python program_synthesis/runner.py --functions fn_a --lengths 50 --attempts 3 --enable-code-interpreter
```

### In-context learning (vLLM)
```bash
python in_context_learning/vllm_incontext.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --functions fn_a --lengths 50 --train-size 200 --test-size 100
```

### Fine-tuning (local training)
```bash
python finetuning/main.py --results-path ./results_new/0/0 --model qwen1.7B --target_func fn_a --sequence_length 50 --train_set_size 200 --test_set_size 10000 --batch_size 20 --n_epochs 200
```


## Repo layout
```
.
├── program_synthesis/        # LLM-PV
├── in_context_learning/      # vLLM few-shot ICL runner (batched inference)
├── finetuning/               # Training/finetuning code + YAML sweeps
├── src/                      # Shared data generators & target functions
├── requirements.txt, environment.yaml
└── README.md
```

> Details, CLI flags, artifacts, and extensions live in each submodule’s README:
> - [`program_synthesis/README.md`](program_synthesis/README.md)  
> - [`in_context_learning/README.md`](in_context_learning/README.md)  
> - [`finetuning/README.md`](finetuning/README.md)
