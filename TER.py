from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the fine-tuned T5 model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def terprocess(text):
    # Example text input
    inputs = tokenizer.encode("emotion: " + text, return_tensors="pt")
    outputs = model.generate(inputs)

    # Decode the output
    emotion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    filename="Emotions.txt"
    with open(filename, "a") as file:
         file.write(f"TER: {emotion}\n")

    
