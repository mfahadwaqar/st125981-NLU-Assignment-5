# NLU Assignment 5: Optimization Human Preference & LLM-as-a-Judge

**Name:** Muhammad Fahad Waqar <br>
**Student ID:** st125981  

---

## Tasks

### Task 1: Dataset Preparation

- **Dataset:** [jondurbin/truthy-dpo-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1)
- A preference dataset designed to teach language models to be truthful and avoid hallucinations
- Each sample contains three fields: `prompt`, `chosen` (factually correct answer), `rejected` (hallucinated answer)
- A 90/10 train/eval split is applied (no predefined test split in the dataset)

### Task 2: DPO Training

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit QLoRA via `BitsAndBytesConfig` (NF4, bfloat16 compute, double quantization)
- **LoRA configuration:** r=64, alpha=128, dropout=0.05, target modules: `q_proj`, `k_proj`, `v_proj`
- **Reference model:** A frozen copy of the base model loaded separately for KL penalty computation

**DPO hyperparameters:**

| Parameter | Value |
|---|---|
| `beta` | 0.1 |
| `learning_rate` | 5e-7 |
| `num_train_epochs` | 3 |
| `max_length` | 512 |
| `max_prompt_length` | 256 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 (effective batch size = 4) |
| `lr_scheduler_type` | linear |
| `warmup_ratio` | 0.1 |
| `bf16` | True |
| `gradient_checkpointing` | True |

Training curves (loss, reward margin, chosen vs rejected rewards) are saved to `training_curves.png`.

### Task 3: Push to Hugging Face Hub

- The trained LoRA adapter is saved locally to `./output-dpo-qwen/` and pushed to Hugging Face Hub
- Only the adapter weights are uploaded (not the full base model)
- **Model URL:** https://huggingface.co/mfahadwaqar20/st125981-qwen2.5-1.5b-dpo-truthy

### Task 4: LLM-as-a-Judge Evaluation

- **Evaluation dataset:** [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) — `helpful_base` subset
- 15 random prompts are sampled (seed=42)
- Responses are generated from both the Base model and the DPO fine-tuned model
- GPT-4o-mini judges each pair of responses using the following prompt template:

> You are a highly qualified and impartial judge evaluating two AI models. Your task is to determine which model provides a better, more accurate, and more helpful response to the user's instruction. [...] Output ONLY your final verdict as exactly one of the following options, with no extra text or explanation: "Model A", "Model B", or "Tie".

**Win Rate formula:**

$$\text{Win Rate} = \frac{\text{Model B Wins} + 0.5 \times \text{Ties}}{\text{Total Valid Evaluations}} \times 100$$

Where Model B is the DPO fine-tuned model. Results are saved to `evaluation_results.png`.

**Final Win Rate: 23.3%**

---

## Results

### Environment

| Item | Value |
|---|---|
| Platform | Kaggle (GPU notebook) |
| GPU | Tesla P100-PCIE-16GB |
| VRAM | 17.1 GB |

### Model & Dataset

| Item | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Tokenizer vocab size | 151,665 |
| Total parameters | 1,555,641,856 |
| LoRA trainable parameters | 11,927,552 (0.77%) |
| Training dataset | jondurbin/truthy-dpo-v0.1 (1,016 samples) |
| Train split | 914 samples (90%) |
| Eval split | 102 samples (10%) |

### Training

| Metric | Value |
|---|---|
| Total optimizer steps | 684 |
| Training time | 63 min 38 sec (3,824 s) |
| Final train loss | 0.6308 |
| Final epoch | ~3.0 |
| Train samples/sec | 0.717 |
| Checkpoints saved | step-600, step-684 |

**Training log (eval steps):**

| Step | Train Loss | Val Loss | Rewards/Chosen | Rewards/Rejected | Rewards/Accuracy | Rewards/Margin |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.6887 | 0.6842 | 0.0079 | -0.0109 | 0.6569 | 0.0188 |
| 200 | 0.6613 | 0.6620 | 0.0196 | -0.0455 | 0.7745 | 0.0651 |
| 300 | 0.6404 | 0.6503 | 0.0158 | -0.0750 | 0.8039 | 0.0908 |
| 400 | 0.6087 | 0.6243 | 0.0265 | -0.1239 | 0.8431 | 0.1503 |
| 500 | 0.6229 | 0.6173 | 0.0312 | -0.1368 | 0.8039 | 0.1681 |
| 600 | 0.5644 | 0.6072 | 0.0297 | -0.1628 | 0.8333 | 0.1925 |

![Training Curves](__results___files/__results___16_0.png)

### Evaluation (LLM-as-a-Judge)

| Metric | Value |
|---|---|
| Evaluation dataset | AlpacaEval `helpful_base` (129 samples total) |
| Prompts sampled | 15 (seed=42) |
| Valid evaluations | 15 |
| Judge model | GPT-4o-mini |
| Model A wins (Base) | 11 |
| Model B wins (DPO) | 3 |
| Ties | 1 |
| Errors | 0 |
| **Win Rate (DPO model)** | **23.3%** |

**Per-sample verdicts:**

| # | Instruction | Winner |
|---|---|---|
| 01 | Why is kobe beef so damn expensive? | Model A |
| 02 | Hi, I'm trying to solve a crossword puzzle, but I've never done one before... | Model A |
| 03 | Please give me a list of planets in our solar system. | Model B |
| 04 | How did serial killers get away with murder for so long in the 70s and 80s? | Tie |
| 05 | I like to host guests at my home from time to time, and I am gathering recipes... | Model B |
| 06 | What is Atlantis? | Model A |
| 07 | What causes the northern lights? | Model A |
| 08 | What type of soil is suitable for cactus? | Model A |
| 09 | I want to learn more about becoming a CPA. How long does it take to become a CPA? | Model A |
| 10 | What are different drawers I should have for clothes? | Model B |
| 11 | Who is Larry Page? | Model A |
| 12 | How do you become an author? | Model A |
| 13 | What are some famous world music artists? | Model A |
| 14 | I've read the book "The Twelve Caesars" by Suetonius. I'm curious about the content... | Model A |
| 15 | What year was the Yamato Battleship built? | Model A |

![Evaluation Results](__results___files/__results___33_1.png)

---

## Results

| Metric | Value |
|---|---|
| DPO Train Loss (final) | 0.6308 |
| Reward Margin (final) | 0.1925 |
| Model B (DPO) Wins | 3 |
| Model A (Base) Wins | 11 |
| Ties | 1 |
| **Win Rate** | **23.3%** |