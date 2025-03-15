import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import time
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# 🔹 Logging functions
def log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

log_progress("🚀 Starting LLaMA 3.2 fine-tuning process...")

try:
    # 🔹 Authentication setup
    log_progress("🔑 Authenticating with Hugging Face...")
    login("ur key")  # Replace with your token
    log_progress("✅ Authentication complete!")

    # 🔹 Model & Tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    log_progress(f"📌 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding
    log_progress("✅ Tokenizer loaded successfully!")

    # 🔹 Quantization settings
    # 🔹 4-bit Quantization settings
    log_progress("⚙️ Configuring model quantization (4-bit)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 🔹 Load model
    log_progress("📥 Loading model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        use_cache=False
    )
    log_progress(f"✅ Model loaded in {time.time() - start_time:.2f} seconds!")

    # 🔹 Prepare for LoRA training
    log_progress("⚙️ Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    log_progress("✅ Model ready for fine-tuning!")

    # 🔹 Load datasets
    datasets = [
        "Amod/mental_health_counseling_conversations",
        "mpingale/mental-health-chat-dataset",
        "heliosbrahma/mental_health_chatbot_dataset"
    ]

    log_progress("📥 Loading and formatting datasets...")
    all_datasets = []

    def clean_text(text):
        """ Remove [INST] ... [/INST], <s>, </s>, and unnecessary formatting. """
        if text is None:  # Prevent NoneType errors
            return ""

        text = re.sub(r"\[INST\](.*?)\[/INST\]", r"\1", text, flags=re.DOTALL)  # Remove instruction markers
        text = re.sub(r"<s>|</s>", "", text)  # Remove <s> and </s>
        
        return text.strip()

    def extract_human_assistant(text):
        """ Extract user and assistant conversation from raw text format. """
        user_match = re.search(r"<HUMAN>[:\s]*(.*?)\n*<ASSISTANT>[:\s]*", text, re.DOTALL)
        assistant_match = re.search(r"<ASSISTANT>[:\s]*(.*)", text, re.DOTALL)

        if not user_match or not assistant_match:
            return None

        return {"input": clean_text(user_match.group(1)), "output": clean_text(assistant_match.group(1))}

    def format_conversation(user_input, assistant_output):
        """ Convert extracted text into LLaMA 3 chat format. """
   
        if not user_input.strip() or not assistant_output.strip():
            return None
        emotion = "neutral"
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a mental health therapy AI assistant.\n\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"[Emotion: {emotion}] {user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_output.strip()}<|eot_id|>"
        )

    for dataset_name in datasets:
        try:
            log_progress(f"📥 Loading dataset: {dataset_name}")
            ds = load_dataset(dataset_name)['train']

            # 🔹 Extract and clean data
            if "Context" in ds.column_names and "Response" in ds.column_names:
                ds = ds.map(lambda e: {"input": clean_text(e["Context"]), "output": clean_text(e["Response"])})
            elif "text" in ds.column_names:
                ds = ds.map(lambda e: extract_human_assistant(e["text"]))
            elif "questionText" in ds.column_names and "answerText" in ds.column_names:
                ds = ds.map(lambda e: {"input": clean_text(e["questionText"]), "output": clean_text(e["answerText"])})
            else:
                log_progress(f"⚠️ Skipping dataset {dataset_name} due to unknown column names.")
                continue

            all_datasets.append(ds)
            log_progress(f"✅ Processed dataset: {dataset_name} (Size: {len(ds)})")

        except Exception as e:
            log_progress(f"❌ Error loading dataset {dataset_name}: {e}")

    combined_dataset = concatenate_datasets(all_datasets)
    log_progress(f"📊 Combined dataset size: {len(combined_dataset)} samples")

    def preprocess_function(examples):
        formatted_texts = [
            format_conversation(clean_text(inp), clean_text(out))
            for inp, out in zip(examples['input'], examples['output'])
        ]
        valid_texts = [text for text in formatted_texts if text is not None]
        if not valid_texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        tokenized = tokenizer(
            valid_texts,
            truncation=True,
            padding='longest',
            max_length=4096,
            return_tensors="pt"
        )

        tokenized['labels'] = tokenized['input_ids'].clone()
        tokenized['labels'][tokenized['labels'] == tokenizer.pad_token_id] = -100
        
        return tokenized

    log_progress("⚙️ Preprocessing dataset...")
    
    tokenized_dataset = combined_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,
        remove_columns=combined_dataset.column_names
    )
    empty_samples = sum(1 for x in tokenized_dataset if len(x['input_ids']) == 0)
    log_progress(f"🚨 Empty samples before filtering: {empty_samples}")
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
    log_progress("✅ Dataset preprocessing complete!")
    valid_sample_count = len(tokenized_dataset)
    print(f"Final samples after filtering: {valid_sample_count}")
    # 🔹 LoRA Configuration
    log_progress("⚙️ Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    log_progress("✅ LoRA applied successfully!")

    # 🔹 Training Arguments
    training_args = TrainingArguments(
        output_dir="./llama-3.2-3b-instruct-finetuned",
        per_device_train_batch_size=1,       # Small due to memory limits
        gradient_accumulation_steps=32,      # 64 is high; 32 balances speed & memory
        num_train_epochs=2,                  # Same as before
        learning_rate=6e-4,                   # Standard for LoRA tuning
        lr_scheduler_type="cosine",           # Good decay schedule
        warmup_ratio=0.01,                    # Slightly lower to prevent slow start
        weight_decay=0.005,                    # More balanced
        fp16=True,                            # Mixed precision training
        gradient_checkpointing=True,          # Reduces memory usage
        max_grad_norm=1.0,                    # More stable than 0.8
        save_total_limit=2,                    # Keep last 2 checkpoints
        save_steps=100,                        # Save frequently for monitoring
        logging_dir="./logs",
        logging_steps=1,                       # Logs every step
        evaluation_strategy="no",              # No eval during training (optional)
        optim="adamw_bnb_8bit",             # Best optimizer for quantized models
        report_to="none"                       # No external logging
    )

    # 🔹 Trainer setup
    log_progress("⚙️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    log_progress("🚀 Starting training...")
    print_gpu_memory()
    trainer.train()
    log_progress("✅ Training complete!")

    model.save_pretrained("./llama-3.2-3b-instruct-finetuned")
    tokenizer.save_pretrained("./llama-3.2-3b-instruct-finetuned")
    log_progress("📥 Model saved successfully!")

except Exception as e:
    log_progress(f"❌ Error occurred: {str(e)}")
    raise

finally:
    log_progress("✅ Process completed!")