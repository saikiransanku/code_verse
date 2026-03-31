from django.shortcuts import render, redirect
from .models import ChatQuery

def home(request):
    if not request.session.session_key:
        request.session.create()
    
    session_key = request.session.session_key
    
    if request.method == "POST":
        prompt = request.POST.get('prompt', '').strip()
        if prompt:
            # Simulated AI Response
            fake_response = f"Simulated AI analysis for: '{prompt}'. (AI integration pending)"
            ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=fake_response)
        return redirect('home')

    history = ChatQuery.objects.filter(session_key=session_key)
    
    return render(request, 'core/index.html', {'history': history})
