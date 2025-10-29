from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('chat_response/', views.chat_response, name='chat_response'),
]