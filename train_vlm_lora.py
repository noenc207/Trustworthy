"""
LoRA Fine-tuning Script cho Qwen2-VL tren anh Da Lieu (Medical VQA).
Yeu cau chay tren Ubuntu co GPU (RTX 3090/4090/5090).
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    print("🚀 Khoi dong kach ban huan luyen LoRA cho Qwen2-VL...")
    
    # 1. Khai bao Model va Dataset
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    train_file = "derma_vqa_train.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ Khong tim thay {train_file}. Vui long chay vqa_dataset_prep.py truoc!")
        return

    # 2. Cau hinh 4-bit (Tiet kiem 70% VRAM)
    print("📦 Dang tai mo hinh duoi dinh dang 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(model_id)

    # 3. Cau hinh LoRA (Chi huan luyen cac lop attention)
    print("🧠 Cau hinh LoRA Adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Chuan bi Dataset
    print("📚 Load dataset...")
    dataset = load_dataset("json", data_files={"train": train_file})
    
    # 5. Cau hinh Huan luyen (SFTTrainer)
    training_args = TrainingArguments(
        output_dir="./checkpoints/qwen2vl_derma_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 16
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=200, # Chay thu 200 buoc de kiem tra
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True, # Dung fp16 cho 3090/5090
        remove_unused_columns=False,
    )

    def formatting_prompts_func(example):
        # Format list cac tin nhan cho Qwen-VL Chat Template
        texts = []
        for msgs in example['messages']:
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return texts

    print("🔥 Bat dau huan luyen (Training)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=lora_config,
        max_seq_length=1024,
        tokenizer=processor.tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
    
    print("💾 Luu mo hinh LoRA (Adapters)...")
    trainer.model.save_pretrained("./checkpoints/qwen2vl_derma_lora_final")
    processor.save_pretrained("./checkpoints/qwen2vl_derma_lora_final")
    print("🎉 HOAN THANH!")

if __name__ == "__main__":
    main()
