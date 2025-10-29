import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import json

genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')

def chatbot(request):
    return render(request, 'chatbot/chatbot.html')

def chat_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message')

        if user_message:
            try:
                response = model.generate_content(user_message)
                bot_response = response.text
                return JsonResponse({'message': bot_response})
            except Exception as e:
                return JsonResponse({'message': f'Lỗi: {e}'})
    return JsonResponse({'message': 'Yêu cầu không hợp lệ'})