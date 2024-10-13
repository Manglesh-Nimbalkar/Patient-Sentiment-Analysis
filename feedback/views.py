from django.shortcuts import render, redirect
from .forms import FeedbackForm
from .models import Feedback
import torch
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import mysql.connector
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
from huggingface_hub import notebook_login

# Prompt to enter your token
notebook_login()

# Define the token variable if not using notebook_login or if running in a script
token = "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"

# Load the sentiment analysis tokenizer and model once, outside the functions
tokenizer_path = 'roberta-base'
model_path = 'feedback/models/roberta.pth'
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
model = RobertaForSequenceClassification.from_pretrained(tokenizer_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Load the emotion model and tokenizer once, outside the functions
emotion_tokenizer = AutoTokenizer.from_pretrained('arpanghoshal/EmoRoBERTa', token=token)
emotion_model = TFAutoModelForSequenceClassification.from_pretrained('arpanghoshal/EmoRoBERTa', token=token)
emotion_pipeline = pipeline('sentiment-analysis', model=emotion_model, tokenizer=emotion_tokenizer, token=token)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_sentiment(sentence):
    cleaned_text = preprocess_text(sentence)
    inputs = tokenizer(cleaned_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    if predicted_label == 1:
        return 'Positive'
    else:
        return 'Negative'

def get_emotion(sentence):
    emotion_dict = {
        'Happy': ['admiration', 'approval', 'caring', 'gratitude', 'joy', 'love', 'optimism', 'relief', 'excitement','amusement', 'curiosity', 'realization'],
        'Anger': ['anger', 'annoyance', 'disappointment', 'disgust'],
        'Fear': ['fear', 'nervousness'],
        'Sad': ['desire', 'disapproval', 'embarrassment', 'grief', 'remorse', 'sadness'],
        'Neutral': [ 'confusion', 'pride', 'neutral']
    }
    
    reaction = emotion_pipeline(sentence)[0]['label']
    for emotion, labels in emotion_dict.items():
        if reaction in labels:
            return emotion

def feedback_form(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data  # Accessing validated form data
            feedback = Feedback.objects.create(
                question_1=cleaned_data.get('question_1'),
                question_2=cleaned_data.get('question_2'),
                question_3=cleaned_data.get('question_3'),
                question_4=cleaned_data.get('question_4'),
                question_5=cleaned_data.get('question_5')
            )
            
            # Process each answer to determine its sentiment
            for i in range(1, 6):
                question_key = f'question_{i}'
                if question_key in cleaned_data and cleaned_data[question_key]:
                    sentiment = predict_sentiment(cleaned_data[question_key])
                    setattr(feedback, f'question_{i}_sentiment', sentiment)
            
            # Combine all answers to find the overall emotion
            all_answers = [cleaned_data[f'question_{i}'] for i in range(1, 6) if cleaned_data.get(f'question_{i}')]
            overall_emotion = get_emotion(" ".join(all_answers))
            feedback.overall_emotion = overall_emotion
            
            # Save the overall emotion in the database
            feedback.emotion = overall_emotion
            feedback.save()
            
            return redirect('thank_you', feedback_id=feedback.id)
    else:
        form = FeedbackForm()
    return render(request, 'feedback/feedback_form.html', {'form': form})

def thank_you(request, feedback_id):
    return render(request, 'feedback/thank_you.html', {'feedback_id': feedback_id})


def fetch_data_from_db():
    # Database connection details
    config = {
        'user': 'root',
        'password': 'manglesh2004',
        'host': 'localhost',
        'database': 'feedback_db'
    }

    # Establishing the connection
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor(dictionary=True)

    # Fetching data from the feedback_feedback table
    cursor.execute("SELECT * FROM feedback_feedback")
    data = cursor.fetchall()

    # Closing the connection
    cursor.close()
    conn.close()

    return data

def analyze_reviews(data):
    sentiments = [
        'question_1_sentiment', 'question_2_sentiment', 'question_3_sentiment', 
        'question_4_sentiment', 'question_5_sentiment'
    ]
    
    sentiment_counts = {question: {"Positive": 0, "Negative": 0, "Not analyzed": 0} for question in sentiments}

    for record in data:
        for question in sentiments:
            sentiment = record.get(question)
            if sentiment in sentiment_counts[question]:
                sentiment_counts[question][sentiment] += 1
    
    result = [[sentiment_counts[question]["Positive"], sentiment_counts[question]["Negative"], sentiment_counts[question]["Not analyzed"]] for question in sentiments]
    
    return result


def view_feedback(request, feedback_id):
    feedback = Feedback.objects.get(id=feedback_id)
    data = fetch_data_from_db()
    
    # Analyze sentiment data for each question
    sentiment_result = analyze_reviews(data)
    
    # Analyze overall emotion counts
    emotion_counts = analyze_emotions(data)
    
    sentiment_data = [
        {'question_number': i, 'sentiment_value': getattr(feedback, f'question_{i}_sentiment')}
        for i in range(1, 6)
    ]

    titles = ["Hospital Staff", "Hygiene Standards", "Communications", "Waiting Time", "Facilities and Amenities"]
    
    context = {
        'feedback': feedback,
        'sentiment_data': sentiment_data,
        'titles': titles,
        'sentiment_result': sentiment_result,
        'emotion_counts': emotion_counts 
    }
    
    return render(request, 'feedback/view_feedback.html', context)


def analyze_emotions(data):
    emotion_counts = {
        'Happy': 0,
        'Anger': 0,
        'Fear': 0,
        'Sad': 0,
        'Neutral': 0,
        'Not analyzed': 0
    }

    for record in data:
        emotion = record.get('emotion', 'Not analyzed')
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts['Not analyzed'] += 1  # Handle unexpected values

    return emotion_counts


# views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from langchain.llms import Ollama

ollama = Ollama(base_url="http://localhost:11434", model="chatbot")

@api_view(['POST'])
def chatbot_response(request):
    user_message = request.data.get('message')
    if user_message:
        response = ollama(user_message)
        return Response({"response": response}, status=200)
    return Response({"error": "No message provided"}, status=400)

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from langchain.llms import Ollama

# Initialize the Ollama model
ollama = Ollama(base_url="http://localhost:11434", model="chatbot")

def home(request):
    return render(request, 'feedback/home.html')

@csrf_exempt
def chatbot_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if not message:
                return JsonResponse({"error": "No message provided"}, status=400)
            
            instruction = "Please respond in less than 15 words."
            full_message = f"{message} {instruction}"
            
            response = ollama.invoke(full_message)
            
            return JsonResponse({"response": response})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def process_chat_emotions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # Load the JSON data
            chat_messages = data.get('chat_messages', [])  # Retrieve chat_messages

            # Ensure chat_messages is a list
            if not isinstance(chat_messages, list):
                return JsonResponse({'status': 'failed', 'message': 'Invalid chat messages format'}, status=400)
            
            # Retrieve the last 3 messages
            last_three_messages = chat_messages[-3:]

            # Combine all elements of the last 3 messages into a single string
            combined_messages = " ".join(last_three_messages)

            # Analyze the emotion of the combined messages
            emotion = get_emotion(combined_messages)

            # Establishing a connection to insert data into the table
            config = {
                'user': 'root',
                'password': 'manglesh2004',
                'host': 'localhost',
                'database': 'feedback_db'
            }
            conn = mysql.connector.connect(**config)
            cursor = conn.cursor()

            # Insert the combined messages and its emotion into the table
            cursor.execute(
                "INSERT INTO chat_emotions (response_text, emotion) VALUES (%s, %s)",
                (combined_messages, emotion)
            )

            # Commit the transaction and close the connection
            conn.commit()
            cursor.close()
            conn.close()

            return JsonResponse({'status': 'success'}, status=200)

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return JsonResponse({'status': 'failed', 'message': 'Database error'}, status=500)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'failed', 'message': 'Invalid JSON'}, status=400)
        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({'status': 'failed', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'failed', 'message': 'Invalid request'}, status=400)
