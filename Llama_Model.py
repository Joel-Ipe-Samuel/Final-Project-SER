import re
from openai import OpenAI
from TTS_Model import text_to_speech

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"  
)

def clean_output(text):
    cleaned_text = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"(){}\[\]-]", "", text)
    return cleaned_text

def chat(user_input, emotion=None):
    # Initialize conversation history
    conversation_history = []

    # If an emotion is provided, append it to the user input for context
    if emotion:
        user_input = f"[Emotion: {emotion}] {user_input}"

    conversation_history.append({"role": "user", "content": user_input})

    model_response = ""
        
    # Make the API call to get a response from the model
    completion = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=conversation_history,
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )
            
    # Process the response from the model
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
            content_chunk = chunk.choices[0].delta.content
            model_response += content_chunk
            cleaned_response = clean_output(content_chunk)
                         
    conversation_history.append({"role": "assistant", "content": model_response})
        
    print("\n")  # For better formatting in output

    # Save the model's response to a file
    file_path = r'Model Response.txt'
    with open(file_path, "a") as file:
        file.write(model_response)
    