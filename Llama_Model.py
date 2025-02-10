from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def chat_with_model(user_input, emotion):
    
    if 'conversation_history' not in globals():
        global conversation_history
        conversation_history = []
        
    # 🔹 Load the fine-tuned model
    model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Use the fine-tuned model directory
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set pad token manually if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=dtype,
    )

    # 🔹 Update conversation history with user input and emotion
    # Add emotion dynamically to user input and structure in model's chat format
    chat = conversation_history + [
        {
            "role": "system",
            "content": "You are a mental health therapy model, hence consider user emotion provided before responding"
        },
        {
            "role": "user",
            "content": f"[Emotion: {emotion}] {user_input.strip()}"
        }
    ]

    # 🔹 Convert chat to model input format using apply_chat_template
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to GPU and ensure attention_mask is set
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 🔹 Generate response with proper padding handling
    outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # Limit the response length
            temperature=0.75,  # Control randomness
            top_p=0.95,  # Nucleus sampling (cumulative probability threshold)
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,  # Padding token
            do_sample=True  # Ensure sampling
        )
        # 🔹 Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if response.startswith("<|begin_of_text|>"):
        response = response[len("<|begin_of_text|>"):]
        
    # The assistant's response will come after the last "<|eot_id|>" and "<|start_header_id|>assistant<|end_header_id|>"
    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0].strip()

    # Update the conversation history with the assistant's response
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    file_path = r'Model Response.txt'
    with open(file_path, "a") as file:
        file.write(assistant_response)

