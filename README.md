# LLM Priors for ERM over Programs

This repository contains three experiment tracks built on shared task generators:
- `program_synthesis`: LLM-PV propose-and-verify program synthesis (OpenAI Responses API)
- `in_context_learning`: local vLLM few-shot inference
- `finetuning`: supervised training and finetuning baselines

## Setup
Recommended:
```bash
conda env create -f environment.yaml
conda activate llm_pv
```

Optional:
```bash
pip install -r requirements.txt
```

Note: `vllm` and `xformers` are intentionally optional and are not installed by default in `requirements.txt`.

## Quickstart
LLM-PV:
```bash
export OPENAI_API_KEY=sk-...
python program_synthesis/runner.py --functions fn_a --lengths 50 --attempts 3
```

In-context learning:
```bash
python in_context_learning/vllm_incontext.py --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --functions fn_a --lengths 50 --train-size 200 --test-size 100
```

Fine-tuning:
```bash
python finetuning/main.py --results-path ./results_new/0/0 --model qwen1.7B --target_func fn_a --sequence_length 50 --train_set_size 200 --test_set_size 10000 --batch_size 20 --n_epochs 200
```

## Repository layout
```text
.
|-- program_synthesis/     # LLM-PV runner and analysis scripts
|-- in_context_learning/   # vLLM ICL experiments
|-- finetuning/            # training and finetuning baselines
|-- src/                   # shared data generators and target functions
|-- requirements.txt
|-- environment.yaml
`-- README.md
```

## Additional docs
- `program_synthesis/README.md`
- `in_context_learning/README.md`
- `finetuning/README.md`
