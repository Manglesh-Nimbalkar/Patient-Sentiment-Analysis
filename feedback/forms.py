# feedback/forms.py

from django import forms

class FeedbackForm(forms.Form):
    question_1 = forms.CharField(
        label='How would you rate the quality of medical care provided by our hospital staff?',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'class': 'feedback-textarea'}),
        required=False
    )
    question_2 = forms.CharField(
        label='How satisfied are you with the cleanliness and hygiene standards maintained in our hospital?',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'class': 'feedback-textarea'}),
        required=False
    )
    question_3 = forms.CharField(
        label='How effective was the communication between you and your healthcare providers during your visit?',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'class': 'feedback-textarea'}),
        required=False
    )
    question_4 = forms.CharField(
        label='How would you rate the waiting time for your appointment or treatment?',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'class': 'feedback-textarea'}),
        required=False
    )
    question_5 = forms.CharField(
        label='How satisfied are you with the facilities and amenities provided by our hospital, such as the waiting area, parking, and cafeteria?',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60, 'class': 'feedback-textarea'}),
        required=False
    )
