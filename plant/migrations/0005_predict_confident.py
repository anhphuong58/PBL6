# Generated by Django 2.2.28 on 2022-12-07 14:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('plant', '0004_auto_20221207_0004'),
    ]

    operations = [
        migrations.AddField(
            model_name='predict',
            name='Confident',
            field=models.FloatField(default=0.9846),
            preserve_default=False,
        ),
    ]
