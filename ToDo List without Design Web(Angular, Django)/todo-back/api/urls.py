from django.urls import path
from api import views

urlpatterns = [
    path('api/task_lists/', views.TaskListList.as_view()),
    path('api/task_lists/<int:pk>/', views.TaskListApi.as_view()),
    path('api/task_lists/<int:pk>/tasks/', views.TasksApi.as_view()),
    path('api/tasks/', views.TasksListApi.as_view()),
    path('api/tasks/<int:pk>/', views.TaskApi.as_view()),
    path('api/users/', views.UserList.as_view()),
    path('api/login/', views.login),
    path('api/logout/', views.logout),
]