from api.models import TaskList, Task
from api.serializers import TaskSerializer, TaskListSerializer2
from api.filters import TaskFilter
from django.http import Http404

# from django.contrib

from django.shortcuts import get_object_or_404

from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework import filters
from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination

from django_filters.rest_framework import DjangoFilterBackend

class TaskListList(generics.ListCreateAPIView):
    queryset = TaskList.objects.all()
    serializer_class = TaskListSerializer2
    permission_classes = (IsAuthenticated, )

    def get_queryset(self):
        return TaskList.objects.filter(created_by=self.request.user)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class TaskListApi(generics.RetrieveUpdateDestroyAPIView):
    queryset = TaskList.objects.all()
    serializer_class = TaskListSerializer2
    permission_classes = (IsAuthenticated, )

class TasksListApi(generics.ListCreateAPIView):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
    permission_classes = (IsAuthenticated, )

class TaskApi(generics.RetrieveUpdateDestroyAPIView):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
    permission_classes = (IsAuthenticated, )

class TasksApi(generics.ListCreateAPIView):
    serializer_class = TaskSerializer
    # pagination_class = LimitOffsetPagination
    permission_classes = (IsAuthenticated, )
    filter_backends = (DjangoFilterBackend,
                       filters.SearchFilter,
                       filters.OrderingFilter,)

    # TODO DjangoFilterBackend
    filterset_fields = ('name', 'created_at')
    filter_class = TaskFilter

    #TODO SearchFilter
    search_fields = ('name', 'created_at')

    #TODO OrderingFilter

    # ordering_fields = ('name', 'created_at')
    ordering = ('created_at',)


    def get_queryset(self):
        tasklist = get_object_or_404(TaskList, id = self.kwargs.get('pk'))
        # try:
        #     tasklist = TaskList.objects.get(id = self.kwargs.get('pk'))
        # except TaskList.DoesNotExist as e:
        #     raise Http404
        queryset = tasklist.tasks

        # name = self.request.query_params.get('name', None)
        # status = self.request.query_params.get('status', None)
        # if name is not None and status is not None:
        #     queryset = queryset.filter(name=name).filter(status=status)
        return queryset
