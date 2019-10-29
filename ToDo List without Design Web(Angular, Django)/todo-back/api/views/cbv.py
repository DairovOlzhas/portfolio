from api.models import TaskList, Task
from api.serializers import TaskSerializer, TaskListSerializer2
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

class TaskListsList(APIView):

    def get(selfs, request):
        task_lists = TaskList.objects.all()
        serializers = TaskListSerializer2(task_lists, many=True)
        return Response(serializers.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = TaskListSerializer2(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TaskListsTasks(APIView):

    def get_objects(self, pk):
        try:
            return TaskList.objects.get(id=pk)
        except TaskList.DoesNotExist as e:
            raise Response(status = status.HTTP_404_NOT_FOUND)

    def get(self, request, pk):
        tasklist = self.get_objects(pk)
        tasks = tasklist.task_set.all()
        serializer = TaskSerializer(tasks, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class TaskApi(APIView):

    def get_object(self, pk):
        try:
            return Task.objects.get(id=pk)
        except Task.DoesNotExist as e:
            raise Response(status=status.HTTP_404_NOT_FOUND)

    def get(self, request, pk):
        task = self.get_object(pk)
        return Response(TaskSerializer(task).data)

    def put(self, request, pk):
        task = self.get_object(pk)
        serializer = TaskSerializer(instance=task, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, pk):
        task = self.get_object(pk)
        task.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
