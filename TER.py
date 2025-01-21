from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned T5 model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def terprocess(text):
    # Example text input
    inputs = tokenizer.encode("emotion: " + text, return_tensors="pt")
    outputs = model.generate(inputs)

    # Decode the output
    emotion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Detected emotion: {emotion}")
