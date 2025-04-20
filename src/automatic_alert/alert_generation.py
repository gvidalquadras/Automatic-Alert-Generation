import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import spacy
from src.ner.ner import NERModel
from src.sa.SA import SentimentClassifier
import fasttext
from torch.nn.utils.rnn import pad_sequence
from typing import List
import predict_ner 
import predict_sa

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Topic classification model
model_name = "cardiffnlp/tweet-topic-21-multi"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Dependency parsing model
nlp = spacy.load("en_core_web_sm")

# Ner model
ft_model = fasttext.load_model("cc.en.300.bin")
ner_model = NERModel(embedding_model=ft_model, hidden_dim=256, num_classes=9).to(device)
ner_model.load_state_dict(torch.load("models/ner_model.pth", map_location=device))

# Sentiment analysis model
# Instanciar el modelo y cargar el state_dict
sa_model = SentimentClassifier()
sa_model.load_state_dict(torch.load("models/SA_model.pth", map_location=device))
sa_model.to(device)
sa_model.eval()

# predefined labels for this topic model (they aren't used)
labels = [
    'arts_&culture', 'business&entrepreneurs', 'celebrity&_pop_culture',
    'diaries_&daily_life', 'family', 'fashion&style', 'film_tv&_video',
    'fitness_&health', 'food&dining', 'gaming', 'learning&_educational',
    'music', 'news_&_social_concern', 'other_hobbies', 'relationships',
    'science_&technology', 'sports', 'travel&_adventure', 'world_news',
    'youth_&_student_life'
]

# topics in a readable format
topics = [
    'Arts & Culture', 'Business & Entrepreneurs', 'Celebrity & Pop Culture',
    'Diaries & Daily Life', 'Family', 'Fashion & Style', 'Film, TV & Video',
    'Fitness & Health', 'Food & Dining', 'Gaming', 'Learning & Educational',
    'Music', 'News & Social Concern', 'Other Hobbies', 'Relationships',
    'Science & Technology', 'Sports', 'Travel & Adventure', 'World News',
    'Youth & Student Life'
    ]

def classify_sentence(text, topics=topics):
    """
    Classify a sentence into one of the predefined topics using a pre-trained model.
    Args:
        text (str): The input sentence to classify.
    Returns:
        str: The predicted topic label.
        float: The probability of the predicted topic.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
    return topics[top_idx], probs[0][top_idx].item()

def extract_ner_entity(ner_tags):
    """
    Extract the first relevant entity from NER-tagged tokens with priority:
    1. B-PER or B-ORG
    2. B-LOC or B-MISC
    """
    # Pass 1: look for B-PER or B-ORG
    for priority_tags in [['B-PER', 'B-ORG'], ['B-LOC', 'B-MISC']]:
        entity_words = []
        capturing = False
        current_type = None

        for word, tag in ner_tags:
            if not capturing:
                if tag in priority_tags:
                    capturing = True
                    entity_words.append(word)
                    current_type = tag.split('-')[1]
            else:
                if tag == f'I-{current_type}':
                    entity_words.append(word)
                else:
                    break  # Stop at the first complete entity

        if entity_words:
            break  # Stop if we found a valid entity

    entity = " ".join(entity_words)
    return entity

def extract_svo(tweet, ner_tags):
    """
    Extract Subject-Verb-Object (SVO) from a tweet using spaCy.
    Args:
        tweet (str): The input tweet text.
        ner_tags (str): The NER tags associated with the tweet.
    Returns:
        tuple: A tuple containing the subject, verb, and object."""
    
    # Extract subject ner entity
    ner_entity = extract_ner_entity(ner_tags)
    
    # Process the tweet with spaCy
    doc = nlp(tweet)

    if ner_entity: 
        sujeto = ner_entity.split()[0]
        verbo = None
        objeto = None
        
        # Iterate through tokens to find SVO
        # Identify subject
        for token in doc:
            if token.text == ner_entity:
                sujeto = ner_entity
            
            # Identify verb
            if token.pos_ == "VERB" and sujeto:
                verbo = token.text
            
            # Identify object
            if token.dep_ in ("obj", "dobj", "obl", "iobj") and verbo:
                objeto = token.text
                break  
            
        return ner_entity, verbo, objeto
    return None, None, None

def generate_alert(tweet): 
    """
    Generate an alert based on the tweet's content.
    Args:
        tweet (str): The input tweet text.
    Returns:
        str: The generated alert message.
    """
    # Classify the tweet
    topic, prob = classify_sentence(tweet)
    
    # Extract NER tags
    tokens = tweet.split()
    ner_tags = predict_ner.predict_ner(ner_model, tokens, device, ft_model, predict_ner.idx2label)
    print(ner_tags)

    # Extract sentiment
    sentiment_label, sentiment_prob = predict_sa.predict_sa(tweet, sa_model, ft_model, device)
    
    # Extract SVO
    subject, verb, obj = extract_svo(tweet, ner_tags)
    
    # Generate alert message
    sentiment = "Good" if sentiment_label == 1 else "Bad"
    alert_message = f"{sentiment} news about {topic}"
    if subject: 
        alert_message += f": {subject} {verb} {obj}."

    
    return alert_message

if __name__ == "__main__":
    # Example tweet
    tweet = "Taylor Swift canceled her concert in London."
    
    # Generate alert
    alert = generate_alert(tweet)

    print(tweet)
    print("Generated Alert:")
    print(alert)




