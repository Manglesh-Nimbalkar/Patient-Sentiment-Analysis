# feedback/models.py

from django.db import models

class Feedback(models.Model):
    question_1 = models.TextField(blank=True, null=True)
    question_2 = models.TextField(blank=True, null=True)
    question_3 = models.TextField(blank=True, null=True)
    question_4 = models.TextField(blank=True, null=True)
    question_5 = models.TextField(blank=True, null=True)

    # Sentiment analysis fields for each question
    question_1_sentiment = models.CharField(max_length=20, default='Not analyzed')
    question_2_sentiment = models.CharField(max_length=20, default='Not analyzed')
    question_3_sentiment = models.CharField(max_length=20, default='Not analyzed')
    question_4_sentiment = models.CharField(max_length=20, default='Not analyzed')
    question_5_sentiment = models.CharField(max_length=20, default='Not analyzed')

    emotion = models.CharField(max_length=50,  default='Not analyzed')

    def __str__(self):
        return f"Feedback {self.id}"

    @property
    def questions_answers_list(self):
        return [
            self.question_1, self.question_2, self.question_3, self.question_4, self.question_5
        ]

    # Method to get sentiment for a specific question
    def get_sentiment(self, question_number):
        field_name = f'question_{question_number}_sentiment'
        return getattr(self, field_name, 'Not analyzed')

class ChatEmotion(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-increment primary key
    response_text = models.CharField(max_length=500, blank=True, null=True)
    emotion = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        db_table = 'chat_emotions'  # Ensure the table name matches the existing table

    def __str__(self):
        return f"ChatEmotion {self.id}"