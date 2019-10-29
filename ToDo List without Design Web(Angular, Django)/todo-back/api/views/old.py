import json
from django.shortcuts import render
from django.http import JsonResponse
from api.models import TaskList, Task
from django.views.decorators.csrf import csrf_exempt
from api.serializers import TaskListSerializer2, TaskSerializer

@csrf_exempt
def task_lists(request):
    if request.method == 'GET':
        task_lists = TaskList.objects.all()
        serializers = TaskListSerializer2(task_lists, many=True)
        return JsonResponse(serializers.data, safe=False)

    elif request.method == 'POST':
        body = json.loads(request.body)
        serializer = TaskListSerializer2(data=body)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors)
    return JsonResponse({'error': 'bad request'})



@csrf_exempt
def task_list(request, pk):
    try:
        task_list = TaskList.objects.get(id=pk)
    except TaskList.DoesNotExist as e:
        return JsonResponse({'error': str(e)})

    if request.method == 'GET':
        return JsonResponse(TaskListSerializer2(task_list).data)

    elif request.method == 'PUT':
        body = json.loads(request.body)
        serializer = TaskListSerializer2(instance=task_list, data=body)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors)

    elif request.method == 'DELETE':
        task_list.delete()
        return JsonResponse({'deleted': True})

    return JsonResponse({'error': 'bad request'})

@csrf_exempt
def tasks(request):
    if request.method == 'GET':
        tasks = Task.objects.all()
        serializer = TaskSerializer(tasks, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        body = json.loads(request.body)
        serializer = TaskSerializer(data=body)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors)
    return JsonResponse({'error': 'bad request'})


@csrf_exempt
def task_lists_tasks(request, pk):
    try:
        task_list = TaskList.objects.get(id=pk)
    except TaskList.DoesNotExist as e:
        return JsonResponse({'error': str(e)})

    if request.method == 'GET':
        tasks = task_list.tasks.all()
        serializer = TaskSerializer(tasks, many=True)
        return JsonResponse(serializer.data, safe=False)
    return JsonResponse({'error': 'bad request'})


@csrf_exempt
def task(request, pk):
    try:
        task = Task.objects.get(id=pk)
    except Task.DoesNotExist as e:
        return JsonResponse({'error': str(e)})

    if request.method == 'GET':
        return JsonResponse(TaskSerializer(task).data)

    elif request.method == 'PUT':
        body = json.loads(request.body)
        serializer = TaskSerializer(instance=task, data=body)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors)

    elif request.method == 'DELETE':
        task.delete()
        return JsonResponse({'deleted': True})
    return JsonResponse({'error': 'bad request'})

