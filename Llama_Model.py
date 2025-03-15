from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import re

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

# 🔹 Phrases that trigger summary generation
END_PHRASES = [
    "please close this chat",
    "thank you for the help i would like to end our conversation",
    "please end this conversation"
]

def generate_summary(conversation_history):
    """
    Generates a structured summary of the conversation history.
    """
    if not conversation_history:
        return "No conversation history available to summarize."

    # Convert list format into readable text
    conversation_text = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history])

    system_prompt = (
        "You are a mental health therapy assistant. Generate a detailed in depth summary followed by structured summary of the following conversation that "
        "can be used by a therapist that highlights key concerns, Feelings expressed, and any other recurring themes.\n\n"
        "**Your response MUST begin with 'Summary:' followed by the summary.**" 
    )
    
    summary_text = f"{conversation_text}"

    # Format the chat for summarization
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summary_text}
    ]

    # Tokenize input
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to GPU

    # 🔹 Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,  # Shorter for summary
        temperature=0.5,  # More controlled output
        top_p=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # 🔹 Decode response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Remove unwanted special characters (like `*`, etc.)
    clean_text = re.sub(r"[*]", "", full_output)

    # Find the LAST occurrence of "Summary:"
    summary_matches = [m.start() for m in re.finditer(r"\bSummary:\s*", clean_text, re.IGNORECASE)]

    if summary_matches:
        # Extract text only from the last "Summary:" onward
        summary = clean_text[summary_matches[-1]:].strip()
    else:
        summary = clean_text  # Fallback in case "Summary:" is missing

    # Save the clean summary
    with open("Therapist_Summary.txt", "w") as file:
        file.write(summary)


def chat_with_model(user_input, emotion):
    global conversation_turns, conversation_history  # Declare globals

    # Check if the input contains an end phrase
    if any(phrase in user_input.lower() for phrase in END_PHRASES):
        summary = generate_summary(conversation_history)
        return f"Session closed. A summary has been generated for therapist reference:\n\n{summary}"

    # 🔹 Normal chat process
    conversation_turns = len(conversation_history) // 2  # Approximate number of turns

    # Adjust system response based on conversation stage
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
        response = response[len("<|begin_of_text|>"):]  # Remove special token if present

    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0].strip()

    # 🔹 Update conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Save response to file
    with open("Model Response.txt", "a") as file:
        file.write(assistant_response + "\n")

