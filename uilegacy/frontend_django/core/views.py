"""
Django Views & Templates - chemai2/frontend_django/core/views.py
Django-based UI with HTMX async updates and API proxying
"""
import json
import requests
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.decorators import login_required
import asyncio
import websockets

# API Gateway configuration
CHEMAI_API_BASE = getattr(settings, 'CHEMAI_API_URL', 'http://localhost:8000/api/v1')

def dashboard(request):
    """Main dashboard view with HTMX template"""
    return render(request, 'core/dashboard.html')

@login_required
def upload_data(request):
    """Handle file upload and forward to FastAPI backend"""
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        # Forward to backend
        try:
            files = {'file': request.FILES['file']}
            data = {'user_id': request.user.id}
            response = requests.post(f"{CHEMAI_API_BASE}/data/upload", files=files, data=data)
            response.raise_for_status()
            return JsonResponse(response.json())
        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': str(e)}, status=502)
    return render(request, 'core/upload.html')

@csrf_exempt
def start_training(request):
    """Start ML training pipeline via backend API"""
    if request.method == 'POST':
        try:
            payload = json.loads(request.body)
            response = requests.post(f"{CHEMAI_API_BASE}/ml/automl", json=payload)
            response.raise_for_status()
            task_id = response.json().get('task_id')
            return JsonResponse({'task_id': task_id, 'status': 'started'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def progress_stream(request, task_id: str):
    """WebSocket/SSE progress streaming endpoint"""
    def event_stream():
        async def subscribe():
            uri = f"ws://localhost:8000/api/v1/ws/progress/{task_id}"
            async with websockets.connect(uri) as ws:
                async for message in ws:
                    yield f"data: {message}\n\n"
        
        loop = asyncio.new_event_loop()
        try:
            for event in loop.run_until_complete(subscribe()):
                yield event
        finally:
            loop.close()
    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
