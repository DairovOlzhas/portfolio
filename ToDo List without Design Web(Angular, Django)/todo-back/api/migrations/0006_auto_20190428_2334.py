# Generated by Django 2.2 on 2019-04-28 17:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_auto_20190428_2308'),
    ]

    operations = [
        migrations.RenameField(
            model_name='task',
            old_name='task_list_id',
            new_name='task_list',
        ),
    ]
