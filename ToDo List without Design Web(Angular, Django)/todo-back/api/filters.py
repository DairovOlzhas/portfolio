from django_filters import rest_framework as filters
from api.models import Task

class TaskFilter(filters.FilterSet):
    name = filters.CharFilter(lookup_expr='exact')
    min_created_at = filters.DateFilter(field_name='created_at', lookup_expr='gte')
    max_created_at = filters.DateFilter(field_name='created_at', lookup_expr='lte')
    class Meta:
        model = Task
        fields = (
            'name',
            'status',
        )
