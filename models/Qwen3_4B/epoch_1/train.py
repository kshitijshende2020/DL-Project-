# train.py — run as: NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 2 train.py
#
# One process per GPU (DDP / true data parallelism). Do NOT pass device_map to
# from_pretrained here — leaving it unset lets each rank load its own full copy
# of the model onto the GPU torchrun assigned it. Passing device_map="auto" (or
# "balanced") instead shards ONE model across GPUs, which is a different
# strategy (model parallelism) and is what was silently happening before.

import os

from unsloth import FastLanguageModel   # import unsloth first, before other ML imports
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
DATA_FILE = "mini_train.parquet"          # path to your parquet dataset
                                           # NOTE: verify this matches the filename gdown actually
                                           # saved in cell 2 (gdown output for a file ID isn't always
                                           # the name you expect) - rename/adjust if needed.
CHAT_TEMPLATE = "qwen-2.5"                # Qwen3 uses the same ChatML-style template
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"
LORA_SAVE_DIR = "qwen3_4b_agri_lora"

SYSTEM_PROMPT = (
    "You are an agricultural extension assistant helping farmers in Kenya "
    "with practical, region-specific farming advice."
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,              # auto-detect: bf16 if supported, else fp16
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

dataset = load_dataset("parquet", data_files=DATA_FILE, split="train")


def to_chatml(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


dataset = dataset.map(to_chatml)


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=8,      # was 3 - you have ~12GB/GPU of headroom on a T4, raise this
                                             # (and/or MAX_SEQ_LENGTH) until VRAM use is where you want it
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        ddp_find_unused_parameters=False,   # required for DDP with LoRA/gradient checkpointing
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

# Only the rank-0 process should write the final adapter, otherwise both
# processes race to write the same files.
if int(os.environ.get("RANK", "0")) == 0:
    model.save_pretrained(LORA_SAVE_DIR)
    tokenizer.save_pretrained(LORA_SAVE_DIR)
    print(trainer_stats)
