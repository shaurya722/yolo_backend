from django.contrib import admin

from api.models import GeneratedVideo, YoloImage

# Register your models here.
admin.site.register(YoloImage)
admin.site.register(GeneratedVideo)