from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# 🔹 Global conversation state
conversation_history = []
conversation_turns = 0  

# 🔹 Load the fine-tuned model once (outside function)
model_id = "meta-llama/Llama-3.2-3B-Instruct"
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

def chat_with_model(user_input, emotion):
    global conversation_turns, conversation_history  # Declare globals

    # Increase turn count
    conversation_turns += 1  

    # Adjust parameters based on conversation stage
    if conversation_turns <= 6:
        system_message = "You are a mental health therapy model. Respond concisely and ask follow-up questions."
    elif 7 >= conversation_turns <= 14:
        system_message = "You are a mental health therapy model. Add a mix of questions and responses."
    else:
        system_message = "You are a mental health therapy model. Provide more detailed responses."

    # 🔹 Update chat context
    chat = conversation_history + [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"[Emotion: {emotion}] {user_input.strip()}"}
    ]

    # Tokenize input
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to GPU

    # 🔹 Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,  # Limit response length
        temperature=0.75,  # Control randomness
        top_p=0.8,  # Nucleus sampling
        repetition_penalty=1.2,  
        pad_token_id=tokenizer.eos_token_id,  
        do_sample=True  
    )

    # 🔹 Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if response.startswith("<|begin_of_text|>"):
        response = response[len("<|begin_of_text|>"):]

    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0].strip()

    # 🔹 Update conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Save response to file
    with open("Model Response.txt", "a") as file:
        file.write(assistant_response + "\n")

    return assistant_response  # Return response for further use
