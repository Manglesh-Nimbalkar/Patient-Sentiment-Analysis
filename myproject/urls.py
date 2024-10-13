# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('feedback.urls')),  # Include 'feedback' app URLs at the root path
]
