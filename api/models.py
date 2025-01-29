from django.db import models

# Create your models here.
class YoloImage(models.Model):
    img = models.ImageField(upload_to='yolo-image/')

class GeneratedVideo(models.Model):
    video = models.FileField(upload_to="annotated_videos/")
    created_at = models.DateTimeField(auto_now_add=True)