import re

# Synonym mapping for emotions (adjust as needed)
emotion_synonyms = {
    "happy": ["joy", "content", "pleasure", "cheerful", "joyful"],
    "sad": ["unhappy", "depressed", "sorrow", "down", "gloomy"],
    "angry": ["mad", "frustrated", "irritated", "enraged","anger"],
    "fearful": ["scared", "afraid", "terrified", "nervous","fear"],
    "surprised": ["shocked", "amazed", "astonished","surprise"],
    "disgusted": ["repulsed", "displeased", "revolted", "nauseated","disgust"],
    "neutral": ["calm", "neutral", "indifferent"]
}

# Normalize emotion (convert to a standard form)
def normalize_emotion(emotion):
    for standard, synonyms in emotion_synonyms.items():
        if emotion.lower() in synonyms or emotion.lower() == standard:
            return standard
    return emotion.lower()  # If no match, return the emotion itself

# Function to read emotions from the file and normalize them
def read_emotions_from_file(filename="Emotions.txt"):
    emotions = {}
    try:
        with open(filename, "r") as file:
            for line in file:
                match = re.match(r"(\w+):\s*(\w+)", line.strip())
                if match:
                    model, emotion = match.groups()
                    emotions[model] = normalize_emotion(emotion)
    except FileNotFoundError:
        print(f"{filename} not found!")
    
    return emotions

# Function to determine the common emotion
def determine_common_emotion(emotions):
    # Count occurrences of each emotion
    emotion_counts = {emotion: list(emotions.values()).count(emotion) for emotion in set(emotions.values())}
    
    # Find the emotion with the highest count
    common_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # If all emotions are different, return the TER emotion as default
    if list(emotion_counts.values()).count(1) == 3:
        common_emotion = emotions.get("TER")
    
    return common_emotion
