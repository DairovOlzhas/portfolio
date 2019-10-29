from django.db import models
from django.contrib.auth.models import User


# class TaskListManager(models.Manager):
#     def for_user(self, user):
#         return self.filter(created_by=user)
#

class TaskList(models.Model):
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, default=None, null=True)

    def __str__(self):
        return '{}: {}'.format(self.name, self.id)


    def to_json(self):
        return {
            'id': self.id,
            'name': self.name
        }

class Task(models.Model):
    name = models.CharField(max_length=200)
    created_at = models.DateField()
    due_on = models.DateField()
    status = models.CharField(max_length=50)
    task_list = models.ForeignKey(TaskList, on_delete=models.CASCADE,
                                  related_name='tasks')


    def __str__(self):
        return '{}: {}'.format(self.name, self.id)


    def to_json(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'due_on': self.due_on,
            'status': self.status,
        }
# Create your models here.
